import torch
from .promise import (
    Promise,
    ensure_dtype
)

class DummyPromise(Promise):
    def __init__(self, promise, traversal_ind, bucket):
        super().__init__(promise, traversal_ind, bucket)

    @property
    def arg(self):
        return self.promise["args"][0]
    
    @property
    def op_result(self):
        return self.arg

    @property
    def rin(self):
        return self.promise["rins"][0]

    @ensure_dtype
    def set_rin(self, new_rin):
        self.promise["rins"][0] = new_rin

    def compute_rins(self):
        """Dummy promise has no logic here"""
        assert self.ready and self.pending_parents == 0, f"Expected Promise {self.id}, {self} to be ready and have 0 pending parents, instead the following state was found: ready: {self.ready}, pending_parents: {self.pending_parents}"        
        self.set_rin(self.rout)
        return self.rout

    @ensure_dtype
    def _setarg(self, value):
        self.promise["args"][0] = value

def compound_promises(promises: list[Promise], traversal_ind, bucket, single_promise_override=False, parent_only_connection=False) -> DummyPromise:
    """Returns a new DummyPromise instance where all input promises are the new
    instance's parents (and it is each of their child).
    The promise dict of the returned instance is only missing arg1 and rin1,
    arg2 and rin2 are set as 0.
    This serves as an aggregator, many-to-one, for promises.
    If only one promise is given, the default behaviour is to return that promise and
    not create a new child promise.
    If single_promise_override=True is given, this behaviour will be overridden and a
    new promise with only the one parent will be created and returned."""
    assert len(promises) > 0, "Empty promises list was given to compound."

    if len(promises) == 1 and not single_promise_override:
        return promises[0]

    p = {
        "rout": torch.zeros_like(promises[0].rout),
        "args": [None],
        "rins": [None],
        "ready": False,
        "complete": False,
        "parents": promises,
    }

    for promise in promises:
        promise.arg_node_ind = traversal_ind

    new_promise = DummyPromise(p, traversal_ind, bucket)

    if not parent_only_connection:
        for promise in promises:
            promise.children.append(new_promise)

    return new_promise