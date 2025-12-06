from .dummy_promise import DummyPromise

class SoftmaxBackwardPromise(DummyPromise):
    
    def __init__(self, promise, traversal_ind, bucket, saved_result, saved_dim):
        super().__init__(promise, traversal_ind, bucket)
        
        # Negative indices -i get saved as 2**32 - i
        if saved_dim > len(self.fwd_shape) - 1:
            saved_dim -= 2**32

        self.dim = saved_dim
        self.result = saved_result.to(self.bucket.dtype)

    @property
    def op_result(self):
        return self.result

    def compute_rins(self):
        """Compute relevance based on AttnLRP"""
        assert self.ready and self.pending_parents == 0, f"Expected Promise {self.id}, {self} to be ready and have 0 pending parents, instead the following state was found: ready: {self.ready}, pending_parents: {self.pending_parents}"
        self.set_rin((self.arg * (self.rout - self.result * self.rout.sum(dim=-1, keepdim=True))))
    
    def refresh_metadata(self, node):
        saved_dim = node._saved_dim
        saved_result = node._saved_result
        # Negative indices -i get saved as 2**32 - i
        if saved_dim > len(self.fwd_shape) - 1:
            saved_dim -= 2**32

        self.dim = saved_dim
        self.result = saved_result
