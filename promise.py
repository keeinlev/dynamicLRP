import torch
from abc import abstractmethod
from torch.autograd.graph import Node
from typing import Callable, Union#, Self
try:
    from typing import Self
except ImportError:
    from typing_extensions import Self
from util import (
    DEBUG,
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
        Promise can be reused by tracking Nodes via index and using the getters on the corresponding Nodes.
    other_branch: if it exists, the branch of the promise that searches for the other argument of the operation
    arg_node_ind: the unique index of the grad_fn node at which this branch's argument was found. For non-leaf Promises,
        i.e. Promises that have children Promises, this is set to the Node where they assigned their child(ren). For
        leaf Promises, this is set to the Node where setarg() gets called.
    arg_node_retrieval_fcn: string attribute name or callable to be gotten/applied on the arg node's grad_fn,
        returns the arg in question
        
    """
    all_inner_nodes : set[int] = set()
    start_nodes_to_promise : dict[int, Self] = {}
    leaf_promises = []
    all_promises = []
    ind_to_node = {}
    def __init__(self, promise, traversal_ind, *args, **kwargs):
        self.promise : dict = promise
        self.parent_idxs : dict[Promise, int] = { p : 0 for p in self.parents }
        self.promise["pending_parents"] = len(self.parents)
        self.children : list[Promise | tuple[Promise]] = [] # Will be added to if further nested promise Nodes are found.

        # These are where the fwd/bwd factory functions will be stored
        self.fwd_list : list[Callable[[Node]]] = []
        self.bwd_list : list[Callable[[Node]]] = []
        # These are where the compiled fwd/bwd chain functions will be stored
        self.compiled_fwd : list[Callable[[torch.Tensor]]] = []
        self.compiled_bwd : list[Callable[[torch.Tensor]]] = []
        self.fwd_shape = promise["rout"].shape # This will update during compilation based on shape-modifying operations
        self.promise["tail_nodes"] = set()
        self.other_branch : Promise = None
        self.arg_node_ind : int = None
        self.arg_node_retrieval_fcn : Union[str, Callable[[Node]]] = None
        self.start_ind : int = traversal_ind
        self.path : set[int] = set()

        if traversal_ind not in Promise.start_nodes_to_promise:
            # This implicitly makes it so that if a Node had a pre-promise associated to it, but also receives
            # a Promise input during normal traversal in the first pass, only the pre-promise is saved for the
            # deterministic pass, since the second promise is only an aggregator, it doesn't store any actual
            # computation.
            # To be clear, the aggregator is still necessary for bringing the args back to a root Promise, but
            # in the deterministic pass, this will happen for all Promises before backward relevance propagation
            # begins, so Promise bwd() execution can be handled manually and aggregator Promises can be skipped.

            # Can access other_branch if needed
            Promise.start_nodes_to_promise[traversal_ind] = self

        if DEBUG:
            Promise.all_promises.append(self)

    @property
    def parents(self):
        return self.promise["parents"]

    @parents.setter
    def parents(self, new_parents):
        self.promise["parents"] = new_parents

    @property
    def pending_parents(self):
        """Returns number of parents of promise which are not complete."""
        return self.promise["pending_parents"]
    
    @pending_parents.setter
    def pending_parents(self, new_pending_parents):
        self.promise["pending_parents"] = new_pending_parents

    @property
    def id(self):
        return (self.start_ind, self.arg_node_ind)

    def add_to_path(self, node_ind):
        self.path.add(node_ind)

    def arg_node_input_index(self, out_adj_list):
        """Returns the index at which this Promise's relevance should land at the arg node."""
        if self.arg_node_ind == self.start_ind:
            if self.parents:
                return self.parents[0].arg_node_input_index(out_adj_list)
            else:
                return 0
        path = sorted(list(self.path.union({self.start_ind, self.arg_node_ind})))
        second_last_node = path[1]
        last_node = self.arg_node_ind
        for (outneighbour, i) in out_adj_list[second_last_node]:
            if outneighbour == last_node:
                return i
        raise ValueError(f"Promise id {self.id} : {self} path resolution could not determine the landing index at the arg node.")

    
    def compile_fwd_bwd(self):
        """The main driver of the input-agnostic specification of this framework.
        Using saved factory functions which take in the autograd node, fetchable from ind_to_node using
        the node's topological index (which is saved with the factory function), assembles a list of
        functions in both the forward and backward directions to execute the Promise chains based on
        the node's metadata in this current run."""
        fwd = []
        if self.fwd_list:
            for node_ind, factory_fcn in self.fwd_list:
                node = Promise.ind_to_node[node_ind] if node_ind is not None else None
                if isinstance(factory_fcn, str) and factory_fcn == "self":
                    fcn = node
                else:
                    fcn = factory_fcn(node)
                
                fwd.append(fcn)

        self.fwd_shape = Promise.ind_to_node[self.arg_node_ind]._input_metadata[0].shape
        self.compiled_fwd = fwd
        
        bwd = []
        if self.bwd_list:
            for node_ind, factory_fcn in self.bwd_list:
                node = Promise.ind_to_node[node_ind] if node_ind is not None else None
                if isinstance(factory_fcn, str) and factory_fcn == "self":
                    fcn = node
                else:
                    fcn = factory_fcn(node)
                bwd.append(fcn)
        
        self.compiled_bwd = bwd
    
    def fwd(self, x):
        for fcn in self.compiled_fwd[::-1]:
            x = fcn(x)
        return x

    def bwd(self, x):
        for fcn in self.compiled_bwd:
            x = fcn(x)
        return x

    def nest_fwd(self, next_f, node_ind):
        """Nests a new operation for recovering the operand for the promise origin.
        Designed to make Promises input-agnostic so they can be reused in later runs.
        
        expects_fwd_shape   : signals that next_f will take an additional positional arg, expected_fwd_shape, which
            can then be used for input-agnostic shape-modifying operations to track the last known shape ahead of the op."""
        self.fwd_list.append((node_ind, next_f))

    def nest_bwd(self, next_f, node_ind):
        """Stacks on a new operation for transforming the promised relevance back down the branch"""
        self.bwd_list.append((node_ind, next_f))
        # most_recent_f = self.bwd
        # self.bwd = lambda x: next_f(most_recent_f(x))
    
    def __add__(self, other: torch.Tensor):
        assert self.fwd_shape == other.shape
        self.nest_bwd(lambda node: (lambda x: x + other), None)
        return self
    
    def clear_args_and_rout(self):
        """Sets all args and rout to None"""
        for i in range(len(self.promise["args"])):
            self.promise["args"][i] = None
        self.set_rout(torch.zeros_like(self.rout))
    
    @classmethod
    def clear_args_and_rout_raw(cls, promise_dict):
        """Sets all args and rout to None for the raw promise dict"""
        for i in range(len(promise_dict["args"])):
            promise_dict["args"][i] = None
        promise_dict["rout"] = torch.zeros_like(promise_dict["rout"])
    
    @classmethod
    def clear_all(cls):
        for p in list(cls.start_nodes_to_promise.values()):
            p.clear_args_and_rout()
            p.promise["rins"] = [ None ] * len(p.promise["rins"])
            p.promise["complete"] = False
            p.promise["ready"] = False
            p.promise["pending_parents"] = len(p.parents)

    @property
    def inner_nodes(self):
        return self.path - set([self.arg_node_ind, self.start_ind])

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
        ...

    @property
    @abstractmethod
    def op_result(self):
        """Returns the forward result of the operation when applied on the promise args"""
        ...

    @property
    def shape(self):
        return self.fwd_shape

    @property
    @abstractmethod
    def rin(self):
        ...

    @property
    def rout(self):
        return self.promise["rout"]

    def set_rout(self, new_rout):
        self.promise["rout"] = new_rout

    @abstractmethod
    def set_rin(self, new_rin):
        ...

    def accumulate_rout(self, new_rout, parent_idx=None):
        assert type(new_rout) == torch.Tensor, f"New rout was not a tensor, but {type(new_rout)}"
        self.promise["rout"] = self.rout + new_rout # Important to do it this way to broadcast

    def exec_bwd(self) -> torch.Tensor:
        """Perform the saved backward execution chain to propagate relevance back down the branch."""
        assert self.ready and self.pending_parents == 0, \
            "Promise backward execution was triggered before promise was ready or before parent promise was complete."
        # assert not self.complete, "Promise is complete, exec_bwd was already called."
        if self.complete:
            return
        res = self.bwd(self.rin)
        self.set_rin(res)

        return res

    @abstractmethod
    def compute_rins(self):
        ...

    def trigger_promise_completion(self, fwd_only=False, recompile=True):
        # This is only called once a promise receives its second argument.
        assert self.ready, "Promise completion was triggered before promise was ready."
        if self.complete:
            return
        if self.pending_parents == 0 and not fwd_only:
            # Either reached root of a promise tree, or we are in the exec_bwd call of a child of a completed promise.
            self.compute_rins()
            res1 = self.exec_bwd()
            all_branch_res = [res1]
            if self.other_branch is not None:
                curr_branch = self.other_branch
                while curr_branch != self:
                    all_branch_res.append(curr_branch.exec_bwd())
                    curr_branch = curr_branch.other_branch

            if self.parents:
                assert (rout := self.rout.nansum()) == (rinsum := sum([ float(r.nansum()) for r in self.promise["rins"] ])) or (rout - rinsum) / self.rout.nansum() < 0.001, \
                    f"Expected child promise to have rout {rout} equal to sum of rins {rinsum}"

            self.set_complete()

            # Now that we have calculated the end relevance_in of all branches, we can feed them to the children promises.
            curr_branch = self
            i = 0
            while (curr_branch != self or i == 0) and curr_branch is not None:
                for child_promise in curr_branch.children:
                    if isinstance(child_promise, tuple): # Branches of the same child promise
                        child_promise[0].pending_parents -= 1
                        child_promise[0].accumulate_rout(all_branch_res[i], child_promise[0].parent_idxs.get(self))
                        if child_promise[0].pending_parents == 0:
                            child_promise[0].trigger_promise_completion()
                    else:
                        child_promise.pending_parents -= 1
                        child_promise.accumulate_rout(all_branch_res[i], child_promise.parent_idxs.get(self))
                        if child_promise.pending_parents == 0:
                            child_promise.trigger_promise_completion()
                curr_branch = curr_branch.other_branch
                i += 1

        else:
            # If there is a parent promise, but it is not complete yet, we can now set its arg with this promise's result.
            # This is what triggers the propagation of the arguments back to the root of the promise tree.
            op_result = self.op_result
            for i, parent in enumerate(self.parents):
                # if self in parent.children: # Edge case for early promise propagation
                parent.promise["tail_nodes"].update(self.promise["tail_nodes"])
                if parent.arg is None:
                    if isinstance(op_result, tuple):
                        parent.setarg(op_result[self.parent_idxs[parent]], fwd_only=fwd_only, recompile=recompile)
                    else:
                        parent.setarg(self.op_result, fwd_only=fwd_only, recompile=recompile)

    @abstractmethod
    def _setarg(self, value):
        ...

    def setarg(self, value, arg_node: torch.autograd.graph.Node = None, ret_fcn: Callable = None, fwd_only = False, recompile=True):
        """Set the corresponding arg for this branch and check if the promise is ready"""
        if arg_node and ret_fcn:
            self.promise["tail_nodes"].add(arg_node)
            self.arg_node_retrieval_fcn = ret_fcn
            Promise.leaf_promises.append(self)
            self.arg_node_ind = arg_node.metadata["topo_ind"]

        # This branch is now terminating, so add the inner nodes to the total set of inner nodes
        Promise.all_inner_nodes.update(self.inner_nodes)

        if recompile:
            # Compile the functions
            self.compile_fwd_bwd()

        if not isinstance(value, float) or value != 0.0:
            self._setarg(self.fwd(value))
        else:
            self._setarg(0.0)
        # self.promise["args"][self.idx] = self.fwd(value)
        if self.set_and_check_ready():
            # print(f"triggering promise {self}")
            self.trigger_promise_completion(fwd_only, recompile)
    
    def retrieve_and_set_new_arg(self, grad_fn):
        if self.arg_node_ind in Promise.start_nodes_to_promise:
            # If the arg node index is a Promise start node, it means it's either the start of a child Promise
            # or that the current Promise only has the one node in its chain
            if self.arg_node_ind != self.start_ind:
                raise ValueError("retrieve_and_set_new_arg should only be called on nodes with leaf Promises, yet it \
                             was found with a child promise.")
        if isinstance(self.arg_node_retrieval_fcn, str):
            attr = getattr(grad_fn, self.arg_node_retrieval_fcn)
            self.setarg(attr, fwd_only=True, recompile=False)
        elif callable(self.arg_node_retrieval_fcn):
            self.setarg(self.arg_node_retrieval_fcn(grad_fn), fwd_only=True, recompile=False)
        else:
            raise ValueError("Saved Promise arg_node_retrieval_fcn was neither a callable nor a string attribute name.")# 

    # def propagate_fwd_shape(self, shape, leaf_promises_accumulator: set):
    #     """Recursive function that should start at the root of a Promise tree and end at the leaf Promises of the tree.
    #     Assumes shape is the correct shape for rout at this Promise on the given run.
    #     This prepares the entire Promise tree for fwd propagation of args."""
    #     self.fwd_shape = shape
    #     self.compile_fwd_bwd() # Will update self.fwd_shape to the end-of-branch shape.

    #     if not self.children:
    #         leaf_promises_accumulator.add(self)

    #     for child in self.children:
    #         if isinstance(child, tuple):
    #             child[0].propagate_fwd_shape(self.fwd_shape, leaf_promises_accumulator)
    #             child[1].propagate_fwd_shape(self.fwd_shape, leaf_promises_accumulator)
    #         else:
    #             child.propagate_fwd_shape(self.fwd_shape, leaf_promises_accumulator)
    
    @classmethod
    def repair_all_parent_child_connections(cls):
        for p in list(cls.start_nodes_to_promise.values()):
            if p.parents:
                for parent in p.parents:
                    if p not in parent.children:
                        parent.children.append(p)

    @classmethod
    def update_all_starting_fwd_shapes(cls):
        """Use the input metadata of the Node at which each Promise starts at to update the fwd_shape member
        at each Promise branch."""
        for node_ind, p in list(cls.start_nodes_to_promise.items()):
            # One branch of each Promise is saved in this lookup
            node : Node = cls.ind_to_node[node_ind]
            
            start_promise : Promise = None
            curr_promise : Promise = p
            while curr_promise != start_promise and curr_promise is not None:
                # Go around the branches cyclically
                if start_promise is None:
                    start_promise = curr_promise
                
                if type(node).__name__ == "CatBackward0":
                    if "shapes" not in node.metadata:
                        # Refresh the shapes of the cat arguments
                        node.metadata["shapes"] = [ out.shape for out in node(torch.rand(node._input_metadata[0].shape)) ]
                    curr_promise.fwd_shape = node.metadata["shapes"][curr_promise.idx]
                else:
                    curr_promise.fwd_shape = node._input_metadata[0].shape
                
                curr_promise.set_rout(torch.zeros(node._input_metadata[0].shape, dtype=curr_promise.rout.dtype, device=curr_promise.rout.device))
                
                curr_promise.compile_fwd_bwd()

                curr_promise = curr_promise.other_branch
