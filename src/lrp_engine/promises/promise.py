import torch
from abc import abstractmethod
from torch.autograd.graph import Node
from typing import Callable, Union
try:
    from typing import Self
except ImportError:
    from typing_extensions import Self

"""General explanation of Promises:
    The idea is that when we come across a Node which requires some forward activation to compute its relevance
    propagation, but the Node did not store this activation in the forward pass, we instead return a "promise",
    a dict wrapped in a class which contains the undistributed relevance, and placeholders for the activations and their
    respective relevances.
    AddBackward0 is one grad_fn that does this, and the most common one, so we will use it as an example.
    So, we return two branches of one Promise, one for each missing operand.
    See that as we pass each branch down the graph, we will encounter one of:
    1. AccumulateGrad or another math-op that we can get the result from
    2. A function that follows the identity or uniform rule like GeluBackward0 or LayerNormBackward0
    3. A mutation function like SliceBackward0 or ReshapeBackward0
    4. (worst case) Another AddBackward0
    For case 2 and 3, we would have to keep an arbitrarily composable function which progressively
    nests the operations that must be done on the result, once it is found, to make it equivalent
    to the downstream addition operand. When we find a node with the result, we simply apply f(result)
    to get the actual operand for the original addition.

        In backward pass:
            Promise start -> op3 -> op2 -> op1 -> result (op0)
            
            See that we need to "bring back" op0's result to the Promise start, and we can do so by
            computing a = op3(op2(op1(result))), yielding the correct operand for the Promise.
            Therefore, for each Node a Promise passes, it stores a copy of that Node's forward function.
    
    If at this time, both operands have been found, compute and store the relevances for both in the
    promise.
    However, once these relevances are computed, notice we are still logically at the Promise start.
    So, we will also need to keep a similar function but for going backwards from the addition back to the
    result node, but this time for the relevance. This is the same argument as above but the opposite direction now.

        Promise start -> op3 -> op2 -> op1 -> op0

        We need to "repropagate" some r_distributed = p(a, r_undistributed) back to op0, where p is characteristic
        to the type of Node that is at the Promise start.
        And see that this can be done by keeping track of every intermediate Node's p functions as well, computing
        r0_undistributed = p1(p2(p3(r_distributed))).
        Therefore, for each Node a Promise passes, it also stores a copy of that Node's relevance propagation function.

    For case 4, we would simply have to nest a promise within the existing promise.
    So the only time this algorithm will branch is if there are multiple of this class of Node with no result-
    yielding grad_fn's in between."""

class PromiseBucket:
    """Holder for all Promises used for a certain model LRP."""
    dtype = torch.float32

    def __init__(self):
        self.all_inner_nodes : set[int] = set()
        self.start_nodes_to_promise : dict[int, Promise] = {}
        self.leaf_promises : list[Promise] = []
        self.ind_to_node = {}
    
    def repair_all_parent_child_connections(self):
        """Ensures that every Promise is linked to any Promises that see it as a child or parent."""
        for p in list(self.start_nodes_to_promise.values()):
            if p.parents:
                for parent in p.parents:
                    if p not in parent.children:
                        parent.children.append(p)

    def update_all_starting_fwd_shapes(self, recompile=True):
        """Use the input metadata of the Node at which each Promise starts at to update the fwd_shape member
        at each Promise branch.
        Used for input-agnostic runs, when shapes of inputs and intermediates possibly change."""
        for node_ind, p in list(self.start_nodes_to_promise.items()):
            # One branch of each Promise is saved in this lookup
            node : Node = self.ind_to_node[node_ind]
            
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

                if hasattr(curr_promise, "refresh_metadata"):
                    curr_promise.refresh_metadata(node)
                
                if recompile:
                    curr_promise.compile_fwd_bwd()

                curr_promise = curr_promise.other_branch

    def clear_all(self):
        """Clears all information in every Promise saved to some start Node"""
        for p in list(self.start_nodes_to_promise.values()):
            p.clear_args_and_rout()
            p.promise["rins"] = [ None ] * len(p.promise["rins"])
            p.promise["complete"] = False
            p.promise["ready"] = False
            p.promise["pending_parents"] = len(p.parents)

def ensure_dtype(func):
    def wrapper(*args, **kwargs):
        args = list(args)
        for i in range(len(args)):
            if isinstance(args[i], torch.Tensor) and args[i].dtype != PromiseBucket.dtype:
                args[i] = args[i].to(PromiseBucket.dtype)
        return func(*args, **kwargs)
    return wrapper

class Promise:
    """
    promise: shared data between the promise origin and both branches.
    parent: parent Promises, i.e. if this promise's result depends on another Promise
    parent_idxs: Used for Promises whose Node accumulates relevance on a position basis, instead of via
        sums. For example, UnbindBackwardPromise, the undistributed relevance is partitioned across the
        unbound dimension.
    children: child Promises, i.e. if this promise will feed its dsitrbuted relevance to other Promises
    fwd_list: stores the required information from every Node along the Promise's path for bringing an activation
        back to the Promise origin.
        Structure: [ (node_topo_ind, next_f_factory), ... ]
            where the order from left to right is the order that the Promise traversed the Nodes in.
            next_f describes a **factory function** which will return the forward function of the Node. It is either
            an enclosed callable or simply "self", if the callable is meant to be the Node corresponding to
            node_topo_ind itself, and can be gotten from Promise.ind_to_node.
    bwd_list: stores the required information from every Node along the Promise's path for propagating the
        distributed relevance from the Promise origin to the arg node.
        Structure: [ (node_topo_ind, next_f_factory), ... ]
            where the order is the same as fwd_list (so they must be called in opposite orders).
            next_f is the same but for the backward relevance propagation functions, can also be "self".
    compiled_fwd: stores the compiled versions of each function in fwd_list, which can be obtained by calling
        each stored factory function with the Node corresponding to the saved index, or retrieving the Node,
        if the factory definition saved is "self"
    compiled_fwd: same thing as compiled_fwd but for the backward relevance propagation functions
    fwd_shape: used as target shape for shape-modifying operations in fwd
    other_branch: if it exists, the branch of the promise that searches for the other argument of the operation
    arg_node_ind: the topo ind of the Node at which this branch's argument was found. For non-leaf Promises,
        i.e. Promises that have children Promises, this is set to the Node where they assigned their child(ren). For
        leaf Promises, this is set to the Node where setarg() gets called.
    arg_node_retrieval_fcn: string attribute name or callable to be gotten/applied on the arg node's grad_fn,
        returns the arg in question
    start_ind: the topo index of the Node at which this Promise was instantiated.
    path: a set of topo indices of all Nodes this Promise has passed by. Does not always include the start or arg nodes.
    """
    def __init__(self, promise, traversal_ind, bucket, *args, **kwargs):
        self.promise : dict = promise
        self.parent_idxs : dict[Promise, int] = { p : 0 for p in self.parents }
        self.promise["pending_parents"] = len(self.parents)
        assert promise["rout"].dtype == PromiseBucket.dtype, f"Promise at node {traversal_ind} had non {PromiseBucket.dtype} rout {promise['rout'].dtype}"
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

        self.bucket : PromiseBucket = bucket

        if traversal_ind not in bucket.start_nodes_to_promise:
            # This implicitly makes it so that if a Node had a pre-promise associated to it, but also receives
            # a Promise input during normal traversal in the first pass, only the pre-promise is saved for the
            # deterministic pass, since the second promise is only an aggregator, it doesn't store any actual
            # computation.
            # To be clear, the aggregator is still necessary for bringing the args back to a root Promise, but
            # in the deterministic pass, this will happen for all Promises before backward relevance propagation
            # begins, so Promise bwd() execution can be handled manually and aggregator Promises can be skipped.

            # Can access other_branch if needed
            bucket.start_nodes_to_promise[traversal_ind] = self

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
        """Returns the index at which this Promise's relevance should land at the arg node.
        Only relevant for Promises that require parent_idxs"""
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

    def nest_fwd(self, next_f_factory, node_ind):
        """Nests a new operation for recovering the operand for the promise origin.
        Designed to make Promises input-agnostic so they can be reused in later runs."""
        self.fwd_list.append((node_ind, next_f_factory))

    def nest_bwd(self, next_f_factory, node_ind):
        """Stacks on a new operation for transforming the promised relevance back down the branch"""
        self.bwd_list.append((node_ind, next_f_factory))
    
    def compile_fwd_bwd(self):
        """The main driver of the input-agnostic specification of this framework.
        Using saved factory functions which take in the autograd node, fetchable from ind_to_node using
        the node's topological index (which is saved with the factory function), assembles a list of
        functions in both the forward and backward directions to execute the Promise chains based on
        the node's metadata in this current run."""
        fwd = []
        if self.fwd_list:
            for node_ind, factory_fcn in self.fwd_list:
                node = self.bucket.ind_to_node[node_ind] if node_ind is not None else None
                if isinstance(factory_fcn, str) and factory_fcn == "self":
                    fcn = node
                else:
                    fcn = factory_fcn(node)
                
                fwd.append(fcn)

        self.fwd_shape = self.bucket.ind_to_node[self.arg_node_ind]._input_metadata[0].shape
        self.compiled_fwd = fwd
        
        bwd = []
        if self.bwd_list:
            for node_ind, factory_fcn in self.bwd_list:
                node = self.bucket.ind_to_node[node_ind] if node_ind is not None else None
                if isinstance(factory_fcn, str) and factory_fcn == "self":
                    fcn = node
                else:
                    fcn = factory_fcn(node)
                bwd.append(fcn)
        
        self.compiled_bwd = bwd
    
    def fwd(self, x):
        """Applies all operations to the operand found from a branch to the origin of the Promise."""
        for fcn in self.compiled_fwd[::-1]:
            x = fcn(x)
        return x

    def bwd(self, x):
        """Applies all operations to the relevance of the operand from the origin of the Promise."""
        for fcn in self.compiled_bwd:
            x = fcn(x)
        return x
    
    def __add__(self, other: torch.Tensor):
        assert self.fwd_shape == other.shape
        self.nest_bwd(lambda node: (lambda x: x + other), None)
        return self

    @property
    def inner_nodes(self):
        """Returns a set of Node indices corresponding to the interior Nodes of the Promise's traversed path."""
        return self.path - set([self.arg_node_ind, self.start_ind])

    @property
    def ready(self):
        """True if all args are set, False otherwise.
        Flags if a Promise is done all forward execution.
        If a Promise is ready and has parents, it will set its parents' arg using its own op_result"""
        return self.promise["ready"]

    def set_and_check_ready(self):
        self.promise["ready"] = all([ x is not None for x in self.promise["args"] ])
        return self.ready
    
    def set_complete(self):
        self.promise["complete"] = True

    @property
    def complete(self):
        """Flags if the promise is done all forward and backward execution.
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

    @ensure_dtype
    def set_rout(self, new_rout):
        self.promise["rout"] = new_rout

    @ensure_dtype
    @abstractmethod
    def _setarg(self, value):
        """Function that actually changes the value of self.promise["args"][self.idx].
        Some Promise types may require unique behaviour."""
        ...

    @ensure_dtype
    @abstractmethod
    def set_rin(self, new_rin):
        """Like _setarg, sets self.promise["rins"][self.idx], but some Promise types need custom behaviour."""
        ...

    @ensure_dtype
    def accumulate_rout(self, new_rout, parent_idx=None):
        assert type(new_rout) == torch.Tensor, f"New rout was not a tensor, but {type(new_rout)}"
        if len(self.rout.shape) == len(new_rout.shape) and self.rout.shape[0] != new_rout.shape[0] and self.rout.shape[1:] == new_rout.shape[1:]:
            # Broadcastable argument shape was incorrectly inferred at Promise creation, fix
            self.promise["rout"] = self.rout.sum(dim=0)
        if self.rout.shape == new_rout.shape[1:]:
            # Broadcastable argument shape was incorrectly inferred for incoming relevance, fix
            new_rout = new_rout.sum(dim=0)
        
        self.promise["rout"] = self.rout + new_rout # Important to do it this way to broadcast from new_rout shape to self.rout shape.

    def exec_bwd(self) -> torch.Tensor:
        """Perform the saved backward execution chain to propagate relevance back down the branch.
        Replaces self.rin with the result of the backprop."""
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
        """Computes the propagation function of the Promise's corresponding Node using self.promise["args"].
        Saves the results in self.promise["rins"]."""
        ...

    def trigger_promise_completion(self, fwd_only=False, recompile=True, accumulate_leaf_promises=None):
        """Triggers the completion of a Promise tree from the bottom-up.
        If the Promise is ready, and it has parents that are not complete, it propagates its op_result as the arg for
        all of its parents, calling parent.setarg(self.op_result), and therefore triggering the completion of its parents
        if they become ready as a result of this (see setarg() for the mutual recursion).
        If the Promise has only complete parents (or no parents), then it can complete its own relevance redistribution
        by calling self.compute_rins(), and propagate its relevance back down the Promise tree by calling self.exec_bwd()
        and accumulating its rin at its child's undistributed relevance (rout), then calling trigger_promise_completion for
        its child."""
        # This is only called once a promise receives its second argument.
        assert self.ready, "Promise completion was triggered before promise was ready."
        if self.complete:
            return
        if self.pending_parents == 0 and not fwd_only:
            # Either reached root of a promise tree, or we are in the exec_bwd call of a child of a completed promise.
            self.compute_rins()
            res1 = self.exec_bwd()
            all_branch_res = [res1]

            # Do the same for all other branches
            if self.other_branch is not None:
                curr_branch = self.other_branch
                while curr_branch != self:
                    all_branch_res.append(curr_branch.exec_bwd())
                    curr_branch = curr_branch.other_branch

            # Check relevance conservation across parent-child link
            # if self.parents:
            #     assert (rout := self.rout.nansum()) == (rinsum := sum([ float(r.nansum()) for r in self.promise["rins"] ])) or (rout - rinsum) / self.rout.nansum() < (0.001 if PromiseBucket.dtype == torch.float32 else 0.01), \
            #         f"Expected child promise to have rout {rout} equal to sum of rins {rinsum}"

            self.set_complete()

            # Now that we have calculated the end relevance_in of all branches, we can feed them to the children promises.
            curr_branch = self
            i = 0
            while (curr_branch != self or i == 0) and curr_branch is not None:
                if not curr_branch.children and accumulate_leaf_promises is not None:
                    accumulate_leaf_promises.append(curr_branch)
                for child_promise in curr_branch.children:
                    # TODO: I think a Promise can only have one child...
                    if isinstance(child_promise, tuple): # Branches of the same child promise
                        child_promise[0].pending_parents -= 1
                        child_promise[0].accumulate_rout(all_branch_res[i], child_promise[0].parent_idxs.get(self))
                        if child_promise[0].pending_parents == 0:
                            child_promise[0].trigger_promise_completion(accumulate_leaf_promises=accumulate_leaf_promises)
                    else:
                        child_promise.pending_parents -= 1
                        child_promise.accumulate_rout(all_branch_res[i], child_promise.parent_idxs.get(self))
                        if child_promise.pending_parents == 0:
                            child_promise.trigger_promise_completion(accumulate_leaf_promises=accumulate_leaf_promises)
                curr_branch = curr_branch.other_branch
                i += 1

        else:
            # If there is a parent promise, but it is not complete yet, we can now set its arg with this promise's result.
            # This is what triggers the propagation of the arguments back to the root of the promise tree.
            op_result = self.op_result
            for i, parent in enumerate(self.parents):
                # Setting the arg of all parents using the op_result of this Promise.
                parent.promise["tail_nodes"].update(self.promise["tail_nodes"])
                if parent.arg is None:
                    if isinstance(op_result, tuple):
                        parent.setarg(op_result[self.parent_idxs[parent]], fwd_only=fwd_only, recompile=recompile)
                    else:
                        parent.setarg(self.op_result, fwd_only=fwd_only, recompile=recompile)

    def setarg(self, value, arg_node: torch.autograd.graph.Node = None, ret_fcn: Callable = None, fwd_only = False, recompile=True):
        """Set the corresponding arg for this branch and check if the promise is ready.
        If it is ready, trigger its completion."""
        if arg_node and ret_fcn:
            self.promise["tail_nodes"].add(arg_node)
            self.arg_node_retrieval_fcn = ret_fcn
            self.bucket.leaf_promises.append(self)
            self.arg_node_ind = arg_node.metadata["topo_ind"]

        # This branch is now terminating, so add the inner nodes to the total set of inner nodes
        self.bucket.all_inner_nodes.update(self.inner_nodes)

        if recompile:
            # Compile the functions
            self.compile_fwd_bwd()

        if isinstance(value, torch.Tensor):
            self._setarg(self.fwd(value))
        else:
            self._setarg(0.0)
        # self.promise["args"][self.idx] = self.fwd(value)
        if self.set_and_check_ready():
            # print(f"triggering promise {self}")
            self.trigger_promise_completion(fwd_only, recompile)
    
    def retrieve_and_set_new_arg(self, grad_fn):
        """Uses arg_node_ind and arg_node_retrieval_fcn to set the arg of a LEAF Promise.
        Used for input-agnostic runs, when saved activations possibly change.
        Does not trigger the backwards relevance propagation once all leaves in a Promise tree have been set."""
        if self.arg_node_ind in self.bucket.start_nodes_to_promise:
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
            raise ValueError("Saved Promise arg_node_retrieval_fcn was neither a callable nor a string attribute name.")
    
    def clear_args_and_rout(self):
        """Sets all args and rout to None"""
        for i in range(len(self.promise["args"])):
            self.promise["args"][i] = None
        self.set_rout(torch.zeros_like(self.rout))
    
    def clear_rins(self):
        for i in range(len(self.promise["rins"])):
            self.promise["rins"][i] = None

    
    @staticmethod
    def clear_args_and_rout_raw(promise_dict):
        """Sets all args and rout to None for the raw promise dict"""
        for i in range(len(promise_dict["args"])):
            promise_dict["args"][i] = None
        promise_dict["rout"] = torch.zeros_like(promise_dict["rout"])
    
