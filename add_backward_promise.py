import torch
from util import epsilon, renormalize_epsilon

class AddBackwardPromise:
    """
    promise: shared data between the promise origin and both branches.
    idx: specifies which argument/operand the branch is looking for.
    parent: parent promise, i.e. if this promise's result depends on another promise
    children: child promises, i.e. if this promise will feed its results to other promises
    fwd: applies all operations to the operand found from a branch to the origin of the promise.
    bwd: applies all operations to the relevance of the operand from the origin of the promise
        to the end of the branch, possibly in steps if one or more Checkpoints were on the branch.
        Structure: [ (checkpoint1, fcn_to_get_from_origin_to_checkpoint1),
                    (checkpoint2, fcn_to_get_from_checkpoint1_to_checkpoint2),
                    ...
                    (None, fcn_to_get_from_last_checkpoint_to_curnode) ]
        So you should apply from left to right, but the inner functions themselves nest right to left.
    fwd_shape: used as target shape for shape-modifying operations in fwd
    other_branch: the branch of the promise that searches for the other argument of the addition
    """
    # all_promises = []
    def __init__(self, promise, idx):
        self.promise = promise
        self.parents : list[AddBackwardPromise] = promise["parents"]
        self.children : list[AddBackwardPromise] = [] # Will be set to list[AddBackwardPromise] if further nested AddBackward Nodes are found.
        self.idx = idx
        self.fwd = lambda x: x
        self.bwd = [ (None, lambda x: x) ] # This will chain in case we come across a Checkpoint partway through
        self.fwd_shape = promise["rout"].shape # This will update after a shape-modifying operation is added to fwd
        self.other_branch : AddBackwardPromise = None
        # self.pending_parents = len(self.parents)
        self.promise["tail_nodes"] = set()

        # AddBackwardPromise.all_promises.append(self) # For Debug

    def nest_fwd(self, next_f):
        """Nests a new operation for recovering the operand for the promise origin"""
        prev_fwd = self.fwd
        self.fwd = lambda x: prev_fwd(next_f(x))

    def checkpoint(self, new_checkpoint):
        """Marks a checkpoint in the backwards op chain and opens a new chain after the checkpoint"""
        self.bwd[-1] = (new_checkpoint, self.bwd[-1][1])
        self.bwd.append((None, lambda x: x))

    def nest_bwd(self, next_f):
        """Stacks on a new operation for transforming the promised relevance back down the branch"""
        last_checkpoint, most_recent_f = self.bwd[-1] # last_checkpoint is actually always None here
        self.bwd[-1] = (last_checkpoint, lambda x: next_f(most_recent_f(x)))
    
    def __add__(self, other: torch.Tensor):
        assert self.fwd_shape == other.shape
        self.nest_bwd(lambda x: x + other)
        return self

    @property
    def pending_parents(self):
        return len([ parent for parent in self.parents if not parent.complete ])

    @property
    def ready(self):
        return self.promise["ready"]

    def set_and_check_ready(self):
        self.promise["ready"] = all([ x is not None for x in self.promise["args"] ])
        return self.ready
    
    def set_complete(self):
        self.promise["complete"] = True

    @property
    def complete(self):
        """Flags if the promise is done all forward and backward execution
        If the promise is complete and has children, it will have set its children's rout value to its bwd(rin)
        result. So the children need only check if parent.complete is True to begin their own exec_bwd()."""
        return self.promise["complete"]

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
    def shape(self):
        return self.fwd_shape

    @property
    def rin(self):
        return self.promise["rins"][self.idx]

    @property
    def rout(self):
        return self.promise["rout"]

    def set_rout(self, new_rout):
        self.promise["rout"] = new_rout

    def accumulate_rout(self, new_rout):
        assert type(new_rout) == torch.Tensor, f"New rout was not a tensor, but {type(new_rout)}"
        self.promise["rout"] = self.rout + new_rout

    def exec_bwd(self) -> tuple[list, torch.Tensor]:
        """Perform each saved backward execution chain to propagate relevance back down the branch.
        Save values for any checkpoints marked along the path and return them with their respective checkpoints."""
        assert self.ready and self.pending_parents == 0, \
            "Promise backward execution was triggered before promise was ready or before parent promise was complete."
        res = self.rin
        checkpoints = []
        for checkpoint, fcn in self.bwd:
            res = fcn(res)
            if checkpoint is not None:
                checkpoints.append((checkpoint, res))

        self.promise["rins"][self.idx] = res

        return checkpoints, res

    def compute_rins(self):
        """Compute base branch relevances based on sum of squares ratios."""
        assert self.ready and self.pending_parents == 0
        arg1, arg2 = self.promise["args"]
        r = self.promise["rout"]
        denom = arg1 ** 2 + arg2 ** 2 + epsilon
        r1 = (arg1 ** 2 / denom) * r
        r2 = (arg2 ** 2 / denom) * r
        self.promise["rins"][0], self.promise["rins"][1] = renormalize_epsilon(r, r1, r2)

    def trigger_promise_completion(self):
        # This is only called once a promise receives its second argument.
        assert self.ready, "Promise completion was triggered before promise was ready."
        if self.pending_parents == 0:
            # Either reached root of a promise tree, or we are in the exec_bwd call of a child of a completed promise.
            self.compute_rins()
            checkpoints, res1 = self.exec_bwd()
            res2 = None
            if self.other_branch is not None:
                checkpoints2, res2 = self.other_branch.exec_bwd()
                checkpoints += checkpoints2
            # Save checkpoint relevances to their grad_fn metadatas to collect later.
            for checkpoint, val in checkpoints:
                checkpoint.metadata["checkpoint_relevance"] = val
            self.set_complete()

            # Now that we have calculated the end relevance_in of this branch, we can feed it to the children promises.
            for child_promise in self.children:
                # if child_promise.rout is None:
                #     child_promise.set_rout(res1.clone())
                # else:
                # child_promise.pending_parents -= 1
                child_promise.accumulate_rout(res1)
                child_promise.trigger_promise_completion()

            if self.other_branch is not None:
                # Do the same for the other branch in this promise. (I should really make Promise and Branch two different classes...)
                for child_promise in self.other_branch.children:
                    # if child_promise.rout is None:
                    #     child_promise.set_rout(res2.clone())
                    # else:
                    # child_promise.pending_parents -= 1
                    child_promise.accumulate_rout(res2)
                    child_promise.trigger_promise_completion()

        else:
            # If there is a parent promise, but it is not complete yet, we can now set its arg with this promise's result.
            # This is what triggers the propagation of the arguments back to the root of the promise tree.
            for parent in self.parents:
                if self in parent.children: # Edge case for early promise propagation
                    parent.promise["tail_nodes"].union(self.promise["tail_nodes"])
                parent.setarg(self.arg1 + self.arg2)

    def setarg(self, value, tail_node: torch.autograd.graph.Node = None):
        """Set the corresponding arg for this branch and check if the promise is ready"""
        if tail_node:
            self.promise["tail_nodes"].add(tail_node)

        self.promise["args"][self.idx] = self.fwd(value)
        if self.set_and_check_ready():
            print(f"triggering promise {self}")
            self.trigger_promise_completion()

def compound_promises(promises: list[AddBackwardPromise], single_promise_override=False, parent_only_connection=False) -> AddBackwardPromise:
    """Returns a new AddBackwardPromise instance where all input promises are the new
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
        "args": [None, 0.0],
        "rins": [None, 0.0],
        "ready": False,
        "complete": False,
        "parents": promises,
    }

    new_promise = AddBackwardPromise(p, 0)

    if not parent_only_connection:
        for promise in promises:
            promise.children.append(new_promise)
    # else:
    #     new_promise.pending_parents = 0

    return new_promise
