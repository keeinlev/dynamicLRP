from abc import abstractmethod
import torch
from util import (
    DEBUG,
    LRPCheckpoint,
)

class Promise:
    """
    promise: shared data between the promise origin and both branches.
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
    other_branch: if it exists, the branch of the promise that searches for the other argument of the operation
    """
    all_promises = []
    def __init__(self, promise, *args, **kwargs):
        self.promise = promise
        self.parents : list[Promise] = promise["parents"]
        self.children : list[Promise | tuple[Promise]] = [] # Will be added to if further nested promise Nodes are found.
        self.fwd = lambda x: x
        self.bwd = [ (None, lambda x: x) ] # This will chain in case we come across a Checkpoint partway through
        self.fwd_shape = promise["rout"].shape # This will update after a shape-modifying operation is added to fwd
        self.promise["tail_nodes"] = set()
        self.other_branch = None

        if DEBUG:
            Promise.all_promises.append(self)

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
        """Returns number of parents of promise which are not complete."""
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
    @abstractmethod
    def arg(self):
        pass

    @property
    @abstractmethod
    def op_result(self):
        """Returns the forward result of the operation when applied on the promise args"""
        pass

    @property
    def shape(self):
        return self.fwd_shape

    @property
    @abstractmethod
    def rin(self):
        pass

    @property
    def rout(self):
        return self.promise["rout"]

    def set_rout(self, new_rout):
        self.promise["rout"] = new_rout

    @abstractmethod
    def set_rin(self, new_rin):
        pass

    def accumulate_rout(self, new_rout):
        assert type(new_rout) == torch.Tensor, f"New rout was not a tensor, but {type(new_rout)}"
        self.promise["rout"] += new_rout

    def exec_bwd(self) -> tuple[list, torch.Tensor]:
        """Perform each saved backward execution chain to propagate relevance back down the branch.
        Save values for any checkpoints marked along the path and return them with their respective checkpoints."""
        assert self.ready and self.pending_parents == 0, \
            "Promise backward execution was triggered before promise was ready or before parent promise was complete."
        # assert not self.complete, "Promise is complete, exec_bwd was already called."
        if self.complete:
            return
        res = self.rin
        checkpoints = []
        for checkpoint, fcn in self.bwd:
            res = fcn(res)
            if checkpoint is not None:
                checkpoints.append((checkpoint, res))

        self.set_rin(res)

        return checkpoints, res

    @abstractmethod
    def compute_rins(self):
        pass

    def trigger_promise_completion(self):
        # This is only called once a promise receives its second argument.
        assert self.ready, "Promise completion was triggered before promise was ready."
        if self.complete:
            return
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
                LRPCheckpoint.save_val(checkpoint, val)
            if self.parents:
                assert (self.rout.nansum() - sum([ float(r.nansum()) for r in self.promise["rins"] ])) / self.rout.nansum() < 0.0001, \
                    f"Expected child promise to have rout {self.rout.nansum()} equal to sum of rins {sum([ float(r.nansum()) for r in self.promise['rins'] ])}"
            self.set_complete()

            # Now that we have calculated the end relevance_in of this branch, we can feed it to the children promises.
            for child_promise in self.children:
                if isinstance(child_promise, tuple): # Branches of the same child promise
                    child_promise[0].accumulate_rout(res1)
                    child_promise[0].trigger_promise_completion()
                else:
                    child_promise.accumulate_rout(res1)
                    child_promise.trigger_promise_completion()


            if self.other_branch is not None and self.other_branch.children:
                # Do the same for the other branch in this promise. (I should really make Promise and Branch two different classes...)
                for child_promise in self.other_branch.children:
                    if isinstance(child_promise, tuple): # Branches of the same child promise
                        child_promise[0].accumulate_rout(res2)
                        child_promise[0].trigger_promise_completion()
                    else:
                        child_promise.accumulate_rout(res2)
                        child_promise.trigger_promise_completion()

        else:
            # If there is a parent promise, but it is not complete yet, we can now set its arg with this promise's result.
            # This is what triggers the propagation of the arguments back to the root of the promise tree.
            for parent in self.parents:
                if self in parent.children: # Edge case for early promise propagation
                    parent.promise["tail_nodes"].union(self.promise["tail_nodes"])
                parent.setarg(self.op_result)

    @abstractmethod
    def _setarg(self, value):
        pass

    def setarg(self, value, tail_node: torch.autograd.graph.Node = None):
        """Set the corresponding arg for this branch and check if the promise is ready"""
        if tail_node:
            self.promise["tail_nodes"].add(tail_node)

        self._setarg(self.fwd(value))
        # self.promise["args"][self.idx] = self.fwd(value)
        if self.set_and_check_ready():
            # print(f"triggering promise {self}")
            self.trigger_promise_completion()
