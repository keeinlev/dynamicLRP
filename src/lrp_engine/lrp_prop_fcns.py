import math
import time
import torch
import torch.nn.functional as F
from functools import reduce
from .promises import *
from .util import (
    epsilon,
    # renormalize_epsilon,
    # renormalize_epsilon_scalar,
    # shift_and_renormalize,
    DEBUG,
    LRPCheckpoint,
    merge_input_shapes,
    gamma_lrp_general,
    epsilon_lrp_matmul,
)
from .relevance_filter import relevance_filter

"""
For all these functions, grad_fn is the autograd Node returned from traversing the autograd graph.
r is the relevance tensor of the output of the given Node.
"""


def output_relevances(func):
    def default_wrapper(*args, **kwargs):
        # Clean input dtypes
        input_rels = args[-1]
        if isinstance(input_rels, tuple):
            input_rels = list(input_rels)
        else:
            input_rels = [ input_rels ]
        for arg in input_rels:
            if isinstance(arg, torch.Tensor) and arg.dtype != PromiseBucket.dtype:
                arg = arg.to(PromiseBucket.dtype)

        res = func(*args, **kwargs)

        # Clean output dtypes
        if isinstance(res, tuple):
            res = list(res)
            for r in res:
                if isinstance(r, torch.Tensor) and LRPPropFunctions != torch.float32 and r.dtype != LRPPropFunctions.dtype:
                    r = r.to(dtype=LRPPropFunctions.dtype)
            res = tuple(res)
            return tuple(res)
        elif isinstance(res, torch.Tensor) and LRPPropFunctions != torch.float32 and res.dtype != LRPPropFunctions.dtype:
            res = res.to(dtype=LRPPropFunctions.dtype)
        return res
    if not DEBUG:
        return default_wrapper
    # # TODO: Needs some refactoring to work properly
    def wrapper(*args, **kwargs):
        res = default_wrapper(*args, **kwargs)
        grad_fn = args[-2]
        r = args[-1]
        print(f"{grad_fn.metadata['topo_ind']}, {grad_fn}:", end="") 
        if isinstance(res, Promise) or type(grad_fn).__name__ == "AccumulateGrad":
            return res
        if (not isinstance(r, Promise)) and \
                not isinstance(res, Promise) \
                and (not isinstance(res, tuple) or not isinstance(res[0], Promise)):
            if isinstance(r, tuple):
                rout = sum(elem.nansum() for elem in r)
            elif isinstance(r, torch.Tensor):
                rout = r.nansum()
            else:
                rout = r
            rins = None
            if isinstance(res, tuple):
                rins = ((res[0].nansum() if isinstance(res[0], torch.Tensor) else res[0]) +
                        (res[1].nansum() if isinstance(res[1], torch.Tensor) else res[1]))
            else:
                rins = res.sum()
            print(rout, rins, end=" ")
        # print(memused := torch.cuda.memory_allocated("cuda:0"), memres := torch.cuda.memory_reserved("cuda:0"), memres - memused)
        print("\n")
        return res
    return wrapper

def skip_redundant_promise(func):
    """Only to decorate Promise-generating prop fcns, like AddBackwardProp, SumBackwardProp, MeanBackward,
    CatBackwardProp, UnbindBackwardProp, etc.
    If a DummyPromise is given as input, and it starts at this Node, reassign its connections to 'skip' over it."""
    def wrapper(*args, **kwargs):
        res = func(*args, **kwargs)
        grad_fn = args[-2]
        r = args[-1]

        if isinstance(r, Promise):
            r = (r,)
        elif isinstance(r, torch.Tensor):
            return res
        elif isinstance(r, tuple) and any(isinstance(elem, torch.Tensor) for elem in r):
            return res
        assert isinstance(r, tuple) and all(isinstance(elem, Promise) for elem in r), f"Error encountered in skip_redundant_promise, expected tuple input of Promises, instead got: {r}"

        # Preprocess both out and in relevances to be tuples of Promises, if applicable.
        if isinstance(res, Promise):
            res = (res,)
        elif isinstance(res, torch.Tensor):
            return res
        elif isinstance(res, tuple) and any(isinstance(elem, torch.Tensor) for elem in res):
            return res
        assert isinstance(res, tuple) and all(isinstance(elem, Promise) for elem in res), f"Error encountered in skip_redundant_promise, expected tuple result of Promises, instead got: {res}"

        for i, in_promise in enumerate(r):
            # Cut out redundant DummyPromise via reassigning links
            if isinstance(in_promise, DummyPromise) and in_promise.start_ind == grad_fn.metadata["topo_ind"]:
                for parent in in_promise.parents:
                    # Reassign all parents to point to children of the DummyPromise and remove the DummyPromise from their children
                    if in_promise in parent.children:
                        parent.children.remove(in_promise)
                        parent.children = list(set(parent.children).union(set(in_promise.children)))
                    res[0].parent_idxs[parent] = i
                # Prop fcns that return Promises will always return branches of the same Promise, so we need
                # only to update the parents via one of the branches.
                res[0].parents.remove(in_promise)
                del res[0].parent_idxs[in_promise]
                res[0].parents = list(set(res[0].parents).union(set(in_promise.parents)))
                in_promise.children = []
                in_promise.parents = []

        # Replace the old Promise with the outputted promise wherever else it is tracked
        res[0].bucket.start_nodes_to_promise[grad_fn.metadata["topo_ind"]] = res[0]
        if "pre_promise" in grad_fn.metadata:
            grad_fn.metadata["pre_promise"] = res[0]

        if len(res) == 1:
            return res[0]
        return res
    return wrapper


def add_node_to_promise_path(func):
    """When a prop fcn is passed any Promise input, adds the curnode's index to each input Promise's path,
    and updates the Promises' fwd_shape to the shape of this Node's forward output."""
    def wrapper(*args, **kwargs):
        r = args[-1]
        grad_fn = args[-2]
        if not isinstance(r, tuple):
            r = (r,)
        for input_ in r:
            if isinstance(input_, Promise):
                input_.add_to_path(grad_fn.metadata["topo_ind"])
                input_.fwd_shape = grad_fn._input_metadata[0].shape
        return func(*args, **kwargs)
    return wrapper
class LRPPropFunctions:

    dtype = torch.float32
    with_grad = False

    @classmethod
    def detach_if_grad(cls, x: torch.Tensor):
        return x if cls.with_grad else x.detach()

    @staticmethod
    @add_node_to_promise_path
    def ResidualIgnoreBackwardProp(grad_fn, r):
        return r

    @staticmethod
    @output_relevances
    @add_node_to_promise_path
    @skip_redundant_promise
    def AddBackwardProp(grad_fn, r):
        """Creates two new AddBackwardPromise objects tied to the same Promise.
        If given a Promise input, the two new objects will be children of the input Promise, and the input
        Promise a parent to the two new objects.
        Otherwise, the relevance is saved as the rout of the new Promise.
        See promise.py for an explanation on Promises."""

        if type(grad_fn.next_functions[1][0]).__name__ == "ResidualIgnoreBackward" and isinstance(r, torch.Tensor):
            return torch.zeros(grad_fn._input_metadata[0].shape, device=grad_fn._input_metadata[0].device, dtype=PromiseBucket.dtype), r

        if grad_fn.next_functions[1][0] is None:
            return r, torch.zeros(grad_fn._input_metadata[0].shape, device=grad_fn._input_metadata[0].device, dtype=PromiseBucket.dtype, requires_grad=LRPPropFunctions.with_grad)
        elif grad_fn.next_functions[0][0] is None:
            return torch.zeros(grad_fn._input_metadata[0].shape, device=grad_fn._input_metadata[0].device, dtype=PromiseBucket.dtype, requires_grad=LRPPropFunctions.with_grad), r

        promise = {
            "rout": r,
            "args": [None, None],
            "rins": [None, None],
            "ready": False,
            "complete": False,
            "parents": [],
        }
        if isinstance(r, Promise):
            promise["parents"] = [r]
            promise["rout"] = torch.zeros(r.fwd_shape, device=r.rout.device, dtype=r.rout.dtype, requires_grad=LRPPropFunctions.with_grad) # Placeholder for shape

        traversal_ind = grad_fn.metadata["topo_ind"]
        bucket = grad_fn.metadata["bucket"]

        promise1 = AddBackwardPromise(promise, traversal_ind, bucket, 0)
        promise2 = AddBackwardPromise(promise, traversal_ind, bucket, 1)

        promise1.other_branch = promise2
        promise2.other_branch = promise1

        if isinstance(r, Promise):
            r.arg_node_ind = traversal_ind
            r.children.append((promise1, promise2))
        
        if (non_residual_child := next((i for i in range(2) if type(grad_fn.next_functions[i][0]).__name__ == "ResidualIgnoreBackward"), None)) is not None:
            promise["non_residual_idx"] = non_residual_child

        grad_fn.metadata["promise"] = promise

        return promise1, promise2
    
    @staticmethod
    @output_relevances
    @add_node_to_promise_path
    @skip_redundant_promise
    def SumBackwardProp(grad_fn, r):
        """Uses Promise structure to correctly propagate relevance back through
        a Sum operation.
        Note that there is both SumBackward0 and SumBackward1.
        0 is for .sum() calls, 1 is for .sum(dim=, keepdim=) calls.
        1 will be triggered as long as `dim` kwarg is given, and `keepdim` is
        an invalid kwarg if `dim` is not also given.
        Therefore we can check for which of 0 or 1 grad_fn is by the presence
        of `_saved_dim` and `_saved_keepdim` attributes."""
        promise = {
            "rout": r,
            "args": [None],
            "rins": [None],
            "ready": False,
            "complete": False,
            "parents": [],
        }
        if isinstance(r, Promise):
            promise["parents"] = [r]
            promise["rout"] = torch.zeros(r.fwd_shape, device=r.rout.device, dtype=r.rout.dtype, requires_grad=LRPPropFunctions.with_grad) # Placeholder for shape
        
        traversal_ind = grad_fn.metadata["topo_ind"]
        bucket = grad_fn.metadata["bucket"]

        is_mean = type(grad_fn).__name__[:-1] == "MeanBackward"
        promise = SumBackwardPromise(promise, traversal_ind, bucket, getattr(grad_fn, "_saved_dim"), getattr(grad_fn, "_saved_keepdim"), is_mean)

        if isinstance(r, Promise):
            r.arg_node_ind = traversal_ind
            r.children.append(promise)

        grad_fn.metadata["promise"] = promise

        return promise
    
    @classmethod
    def MeanBackwardProp(cls, grad_fn, r):
        """Mean is just a scaled sum by 1/n, so the ratios of all elements
        and their cls, contributions should still be the same as if they were a normal
        sum."""
        return cls.SumBackwardProp(grad_fn, r)
    
    @staticmethod
    @output_relevances
    @add_node_to_promise_path
    @skip_redundant_promise
    def CatBackwardProp(grad_fn, r):
        if isinstance(r, Promise):
            rout_placeholder = torch.zeros(r.fwd_shape, device=r.rout.device, dtype=r.rout.dtype, requires_grad=LRPPropFunctions.with_grad)
            shapes = [ out.shape for out in grad_fn(rout_placeholder) ]
            num_args = len(shapes)
            promise = {
                "rout": rout_placeholder,
                "args": [ None for _ in range(num_args) ],
                "rins": [ None for _ in range(num_args) ],
                "ready": False,
                "complete": False,
                "parents": [r],
            }
            
            traversal_ind = grad_fn.metadata["topo_ind"]
            bucket = grad_fn.metadata["bucket"]

            prev_promise_branch = None
            promise_branches = []
            # Since you can cat an arbitrary number of tensors, we need to get a bit creative and turn other_branch into a cyclic connection
            for i in range(num_args):
                new_branch = CatBackwardPromise(promise, traversal_ind, bucket, grad_fn._saved_dim, i)
                new_branch.fwd_shape = shapes[i]
                promise_branches.append(new_branch)
                if prev_promise_branch is not None:
                    prev_promise_branch.other_branch = new_branch
                prev_promise_branch = new_branch
            
            # Create the last-first connection
            if num_args > 1:
                promise_branches[-1].other_branch = promise_branches[0]

            r.arg_node_ind = traversal_ind
            r.children.append(tuple(promise_branches))

            grad_fn.metadata["promise"] = promise

            return tuple(promise_branches)
        
        return grad_fn(r)
    
    @staticmethod
    @output_relevances
    @add_node_to_promise_path
    @skip_redundant_promise
    def StackBackwardProp(grad_fn, r):
        if isinstance(r, Promise):
            unstacked_shape = list(r.fwd_shape)
            stack_dim = grad_fn._saved_dim
            num_args = unstacked_shape[stack_dim]
            unstacked_shape = tuple(unstacked_shape[:stack_dim] + unstacked_shape[stack_dim + 1:])
            promise = {
                "rout": torch.zeros(r.fwd_shape, device=r.rout.device, dtype=r.rout.dtype, requires_grad=LRPPropFunctions.with_grad),
                "args": [ None for _ in range(num_args) ],
                "rins": [ None for _ in range(num_args) ],
                "ready": False,
                "complete": False,
                "parents": [r],
            }
            
            traversal_ind = grad_fn.metadata["topo_ind"]
            bucket = grad_fn.metadata["bucket"]

            prev_promise_branch = None
            promise_branches = []
            # Same cyclic connections as CatBackwardPromise
            for i in range(num_args):
                new_branch = CatBackwardPromise(promise, traversal_ind, bucket, stack_dim, i)
                new_branch.fwd_shape = unstacked_shape
                promise_branches.append(new_branch)
                if prev_promise_branch is not None:
                    prev_promise_branch.other_branch = new_branch
                prev_promise_branch = new_branch
            
            # Create the last-first connection
            if num_args > 1:
                promise_branches[-1].other_branch = promise_branches[0]

            r.arg_node_ind = traversal_ind
            r.children.append(tuple(promise_branches))

            grad_fn.metadata["promise"] = promise

            return tuple(promise_branches)
        
        return grad_fn(r)
    
    @staticmethod
    @output_relevances
    @add_node_to_promise_path
    @skip_redundant_promise
    def UnbindBackwardProp(grad_fn, r):
        # r is a list of inputs, with positions corresponding to the grad_fn inputs.
        if any(isinstance(elem, Promise) for elem in r):
            dtype = parents[0].rout.dtype
            device = parents[0].rout.device
            parents = [ (i, elem) for i, elem in enumerate(r) if isinstance(elem, Promise) ]
            promise = {
                "rout": None,
                "args": [None],
                "rins": [None],
                "ready": False,
                "complete": False,
                "parents": [ t[1] for t in parents ],
            }

            traversal_ind = grad_fn.metadata["topo_ind"]
            bucket = grad_fn.metadata["bucket"]

            p = UnbindBackwardPromise(promise, traversal_ind, bucket, grad_fn._saved_dim)
            for i, parent in parents:
                p.parent_idxs[parent] = i
                parent.arg_node_ind = traversal_ind

            if any(elem is None for elem in r):
                # We don't want to accumulate any tensor rout until we come back around in NTM, for consistency (see lrp.py line 267).
                grad_fn.metadata["pre_promise"] = p
                promise["rout"] = torch.zeros(merge_input_shapes(grad_fn), dtype=dtype, device=device, requires_grad=LRPPropFunctions.with_grad)
            else:
                # If we are processing for the first time in NTM anyway, then it's fine.
                for i, elem in r:
                    if isinstance(elem, torch.Tensor):
                        p.accumulate_rout(elem, i)

            return p
        else:
            return grad_fn(*r)
    
    @staticmethod
    @output_relevances
    @add_node_to_promise_path
    @skip_redundant_promise
    def SplitBackwardProp(grad_fn, r):
        # r is a list of inputs, with positions corresponding to the grad_fn inputs.
        if any(isinstance(elem, Promise) for elem in r):
            dtype = parents[0].rout.dtype
            device = parents[0].rout.device
            parents = [ (i, elem) for i, elem in enumerate(r) if isinstance(elem, Promise) ]
            promise = {
                "rout": None,
                "args": [None],
                "rins": [None],
                "ready": False,
                "complete": False,
                "parents": [ t[1] for t in parents ],
            }

            traversal_ind = grad_fn.metadata["topo_ind"]
            bucket = grad_fn.metadata["bucket"]

            split_size = getattr(grad_fn, "_saved_split_size", None)
            p = SplitBackwardPromise(promise, traversal_ind, bucket, grad_fn._saved_dim, split_size)
            for i, parent in parents:
                p.parent_idxs[parent] = i
                parent.arg_node_ind = traversal_ind

            if any(elem is None for elem in r):
                # We don't want to accumulate any tensor rout until we come back around in NTM, for consistency (see lrp.py line 267).
                grad_fn.metadata["pre_promise"] = p
                promise["rout"] = torch.zeros(merge_input_shapes(grad_fn), dtype=dtype, device=device, requires_grad=LRPPropFunctions.with_grad)
            else:
                # If we are processing for the first time in NTM anyway, then it's fine.
                for i, elem in r:
                    if isinstance(elem, torch.Tensor):
                        p.accumulate_rout(elem, i)

            return p
        else:
            return grad_fn(*r)

    @staticmethod
    @output_relevances
    @add_node_to_promise_path
    def MaxBackwardProp(grad_fn, r):
        max_type = 0 if hasattr(grad_fn, "_saved_dim") else 1

        if isinstance(r, Promise):
            if max_type == 0:
                r.setarg(grad_fn._saved_result, grad_fn, "_saved_result")
                if r.complete:
                    r = r.rin
                else:
                    return r
            else:
                def factory_fcn(node):
                    def fwd_max(x):
                        return torch.max(x, dim=node._saved_dim, keepdim=node._saved_keepdim)
                    return fwd_max

                r.nest_fwd(factory_fcn, grad_fn.metadata["topo_ind"])
                r.nest_bwd("self", grad_fn.metadata["topo_ind"])
                return r

        return grad_fn(r) # Reuse autograd implementation
    
    @staticmethod
    @output_relevances
    @add_node_to_promise_path
    def MaskedFillBackwardProp(grad_fn, r):
        # TODO: need a promise for when the fill value is gradient-tracked
        if isinstance(r, Promise):
            def fwd_factory(node):
                mask = node._saved_mask
                def fwd_reshape(x: torch.Tensor):
                    return x.masked_fill(mask, float('-inf'))
                return fwd_reshape

            r.nest_fwd(fwd_factory, grad_fn.metadata["topo_ind"])
            r.nest_bwd("self", grad_fn.metadata["topo_ind"])
            return r
        return grad_fn(r)
    
    @staticmethod
    @output_relevances
    @add_node_to_promise_path
    def GatherBackwardProp(grad_fn, r):
        if isinstance(r, Promise):
            def fwd_gather(node):
                saved_dim = node._saved_dim
                if saved_dim > len(node._saved_self.shape) - 1:
                    saved_dim -= 2**32
                return torch.gather(node._saved_self, saved_dim, node._saved_index)
            r.setarg(fwd_gather(grad_fn), grad_fn, fwd_gather)
            if r.complete:
                r = r.rin
            else:
                return r
        return grad_fn(r)

    @staticmethod
    @output_relevances
    @add_node_to_promise_path
    def ViewBackwardProp(grad_fn, r):
        upstream_shape = grad_fn._saved_self_sym_sizes
        if isinstance(r, Promise):

            def fwd_factory(node):
                expected_fwd_shape = node._input_metadata[0].shape
                def fwd_reshape(x: torch.Tensor):
                    try:
                        return x.view(expected_fwd_shape)
                    except RuntimeError:
                        return torch.reshape(x, expected_fwd_shape)
                return fwd_reshape
            
            def bwd_factory(node):
                def bwd_reshape(x: torch.Tensor):
                    try:
                        return x.view(node._saved_self_sym_sizes)
                    except RuntimeError:
                        return torch.reshape(x, node._saved_self_sym_sizes)
                return bwd_reshape
            
            r.nest_fwd(fwd_factory, grad_fn.metadata["topo_ind"])
            r.nest_bwd(bwd_factory, grad_fn.metadata["topo_ind"])
            return r
        return r.reshape(upstream_shape)

    @classmethod
    def UnsafeViewBackwardProp(cls, grad_fn, r):
        return cls.ViewBackwardProp(grad_fn, r)

    @classmethod
    def ReshapeBackwardProp(cls, grad_fn, r):
        return cls.ViewBackwardProp(grad_fn, r)

    @classmethod
    def ReshapeAliasBackwardProp(cls, grad_fn, r):
        return cls.ViewBackwardProp(grad_fn, r)

    @staticmethod
    @output_relevances
    @add_node_to_promise_path
    def SliceBackwardProp(grad_fn, r):
        def get_clean_start_end(node):
            expected_fwd_shape = node._input_metadata[0].shape
            upstream_shape = node._saved_self_sym_sizes
            sliced_dim = node._saved_dim
            start = node._saved_start # TODO: Come back to take care of the negative index case.
            full_size = upstream_shape[sliced_dim]
            if start > full_size:
                start = full_size - (2**32 - start)
            end = start + expected_fwd_shape[sliced_dim]
            return start, end, full_size

        def create_pad(node):
            upstream_shape = node._saved_self_sym_sizes
            sliced_dim = node._saved_dim
            start, end, full_size = get_clean_start_end(node)

            # We wish to pad r so that it becomes the correct size along the sliced dimension
            pad = []
            to_pad = [start, full_size - end]

            # Iterate in reverse order, since F.pad() takes in dims from last to first,
            # see https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.pad.html
            # All dims other than sliced_dim should be 0, 0
            for dim in range(len(upstream_shape) - 1, -1, -1):
                pad += [0, 0] if dim != sliced_dim else to_pad

            return tuple(pad)

        if isinstance(r, Promise):
            def fwd_factory(node):
                start, end, _ = get_clean_start_end(node)
                def fwd_slice(x):
                    return torch.ops.aten.slice(x, node._saved_dim, start, end)
                return fwd_slice
            
            def bwd_factory(node):
                def bwd_slice(x):
                    return F.pad(x, create_pad(node))
                return bwd_slice

            r.nest_fwd(fwd_factory, grad_fn.metadata["topo_ind"])
            r.nest_bwd("self", grad_fn.metadata["topo_ind"])
            return r
        return F.pad(r, create_pad(grad_fn))

    @staticmethod
    @output_relevances
    @add_node_to_promise_path
    def IndexBackwardProp(grad_fn, r):
        if isinstance(r, Promise):
            def fwd_factory(node):
                def fwd_index(x):
                    return torch.ops.aten.index(x, [ torch.tensor(x) if x is not None else None for x in node._saved_indices ])
                return fwd_index

            r.nest_fwd(fwd_factory, grad_fn._metadata["topo_ind"])
            r.nest_bwd("self", grad_fn.metadata["topo_ind"])
            return r
        return grad_fn(r)
    
    @staticmethod
    def SimpleShapeBackwardProp(grad_fn, r):
        if isinstance(r, Promise):
            r.nest_fwd("self", grad_fn.metadata["topo_ind"])
            r.nest_bwd("self", grad_fn.metadata["topo_ind"])
            return r

        return grad_fn(r)

    @staticmethod
    @output_relevances
    @add_node_to_promise_path
    def SelectBackwardProp(grad_fn, r):

        def undoSelect(node, x):
            upstream_shape = node._saved_self_sym_sizes
            dim = grad_fn._saved_dim
            idx = node._saved_index
            out = torch.zeros(upstream_shape, dtype=x.dtype, device=x.device, requires_grad=LRPPropFunctions.with_grad)
            out.select(dim, idx).copy_(x)
            return out

        if isinstance(r, Promise):
            def fwd_factory(node):
                def fwd_select(x):
                    return torch.select(x, node._saved_dim, node._saved_index)
                return fwd_select

            r.nest_fwd(fwd_factory, grad_fn.metadata["topo_ind"])
            r.nest_bwd("self", grad_fn.metadata["topo_ind"])
            return r

        return grad_fn(r)

    @classmethod
    @output_relevances
    @add_node_to_promise_path
    def TBackwardProp(cls, grad_fn, r):
        # Not sure why TBackward is different from TransposeBackward, but it seems like this is only
        # in Linear layer matmuls on W for xW^T before Mm and Addmm operations.
        # assert len(r.shape) == 2, "Assumption was that matrix would be 2d Linear weights." # For now assume that it is only 2d matmuls for Linear layers.
        return cls.SimpleShapeBackwardProp(grad_fn, r)

    @classmethod
    @output_relevances
    @add_node_to_promise_path
    def TransposeBackwardProp(cls, grad_fn, r):
        return cls.SimpleShapeBackwardProp(grad_fn, r)

    @staticmethod
    @output_relevances
    @add_node_to_promise_path
    def PermuteBackwardProp(grad_fn, r):
        if isinstance(r, Promise):

            def fwd_factory(node):
                def fwd_permute(x: torch.Tensor):
                    return x.permute(node._saved_dims)
                return fwd_permute

            r.nest_fwd(fwd_factory, grad_fn.metadata["topo_ind"])
            r.nest_bwd("self", grad_fn.metadata["topo_ind"])
            return r
        return grad_fn(r)

    @staticmethod
    @output_relevances
    @add_node_to_promise_path
    def ExpandBackwardProp(grad_fn, r):

        def fwd_factory(node):
            expected_fwd_shape = node._input_metadata[0].shape
            upstream_shape = node._saved_self_sym_sizes
            downstream_shape = expected_fwd_shape
            assert len(upstream_shape) == len(downstream_shape), "Expand should not increase number of dimensions."
            expand_input = [ dim2 if dim1 != dim2 else -1 for dim1, dim2 in zip(upstream_shape, downstream_shape) ]

            def fwd_expand(x):
                return x.expand(*expand_input)

            return fwd_expand

        if isinstance(r, Promise):
            r.nest_fwd(fwd_factory, grad_fn.metadata["topo_ind"])
            r.nest_bwd("self", grad_fn.metadata["topo_ind"])
            return r

        return grad_fn(r)
    
    @classmethod
    @output_relevances
    @add_node_to_promise_path
    def RepeatBackwardProp(cls, grad_fn, r):

        def fwd_factory(node):
            repeats = node._saved_repeats
            def fwd_repeat(x):
                return x.repeat(*repeats)
            return fwd_repeat

        if isinstance(r, Promise):
            r.nest_fwd(fwd_factory, grad_fn.metadata["topo_ind"])
            r.nest_bwd("self", grad_fn.metadata["topo_ind"])
            return r

        return grad_fn(r)
    
    @classmethod
    @output_relevances
    @add_node_to_promise_path
    def UnsqueezeBackwardProp(cls, grad_fn, r):

        def fwd_factory(node):
            dim = node._saved_dim
            def fwd_unsqueeze(x):
                return x.unsqueeze(dim=dim)
            return fwd_unsqueeze

        if isinstance(r, Promise):
            r.nest_fwd(fwd_factory, grad_fn.metadata["topo_ind"])
            r.nest_bwd("self", grad_fn.metadata["topo_ind"])
            return r

        return grad_fn(r)
    
    @classmethod
    @output_relevances
    @add_node_to_promise_path
    def SqueezeBackwardProp(cls, grad_fn, r):

        def fwd_factory(node):
            dim = node._saved_dim
            def fwd_squeeze(x):
                return x.squeeze(dim=dim)
            return fwd_squeeze

        if isinstance(r, Promise):
            r.nest_fwd(fwd_factory, grad_fn.metadata["topo_ind"])
            r.nest_bwd("self", grad_fn.metadata["topo_ind"])
            return r

        return grad_fn(r)
    
    @staticmethod
    @output_relevances
    @add_node_to_promise_path
    def NegBackwardProp(grad_fn, r):
        return r
    
    @staticmethod
    @output_relevances
    @add_node_to_promise_path
    def CloneBackwardProp(grad_fn, r):
        return r
    
    @staticmethod
    @output_relevances
    @add_node_to_promise_path
    def ToCopyBackwardProp(grad_fn, r):
        return r

    @staticmethod
    @output_relevances
    @add_node_to_promise_path
    def MulBackwardProp(grad_fn, r):
        arg1 : torch.Tensor = grad_fn._saved_self
        arg2 : torch.Tensor = grad_fn._saved_other

        if isinstance(r, Promise):
            if arg1 is None:
                def fwd_factory(node):
                    def fwd_mul(x):
                        return x * node._saved_other
                    return fwd_mul
                r.nest_fwd(fwd_factory, grad_fn.metadata["topo_ind"])
            else:
                r.setarg(LRPPropFunctions.detach_if_grad(arg1) * LRPPropFunctions.detach_if_grad(arg2), grad_fn, lambda fcn: LRPPropFunctions.detach_if_grad(fcn._saved_self) * LRPPropFunctions.detach_if_grad(fcn._saved_other))
                if r.complete:
                    r = r.rin
                else:
                    return r # We will check if the promise is complete in the graph traversal

        if arg1 is None:
            # Tensor-scalar product, disregard scalar
            return r, 0.0
        
        arg1, arg2 = LRPPropFunctions.detach_if_grad(arg1), LRPPropFunctions.detach_if_grad(arg2)

        denom = arg1.abs() + arg2.abs() + epsilon

        # Reduce relevance to denom’s shape instead of expanding denom to r
        factor = r.sum_to_size(*denom.shape)  # requires PyTorch ≥ 2.0
        r1 = (arg1.abs() / denom) * factor
        r2 = (arg2.abs() / denom) * factor

        # r1, r2 = renormalize_epsilon(r, r1, r2)
        return r1, r2

    @staticmethod
    @output_relevances
    @add_node_to_promise_path
    def DivBackwardProp(grad_fn, r):
        arg1 = grad_fn._saved_self
        arg2 = grad_fn._saved_other

        if isinstance(r, Promise):
            if arg1 is None:
                def fwd_factory(node):
                    def fwd_div(x):
                        return x / node._saved_other
                    return fwd_div
                r.nest_fwd(fwd_factory, grad_fn.metadata["topo_ind"])
            else:
                r.setarg(LRPPropFunctions.detach_if_grad(arg1) / LRPPropFunctions.detach_if_grad(arg2), grad_fn, lambda fcn: LRPPropFunctions.detach_if_grad(fcn._saved_self) / LRPPropFunctions.detach_if_grad(fcn._saved_other))
                if r.complete:
                    r = r.rin
                else:
                    return r # We will check if the promise is complete in the graph traversal

        if arg1 is None:
            # Tensor-scalar product, disregard scalar
            return r, 0.0
        
        arg1, arg2 = LRPPropFunctions.detach_if_grad(arg1), LRPPropFunctions.detach_if_grad(arg2)

        denom = arg1.abs() + (1 / arg2).abs() + epsilon
        r1 = (arg1.abs() / denom) * r
        r2 = ((1 / arg2).abs() / denom) * r

        # r1, r2 = renormalize_epsilon(r, r1, r2)
        # print(f"DivBackward relevance sums: out {r.sum()}, in {(r1 + r2).sum()}")

        return r1, r2

    @staticmethod
    @output_relevances
    @add_node_to_promise_path
    def MmBackwardProp(grad_fn, r):
        x = LRPPropFunctions.detach_if_grad(grad_fn._saved_self) # s d
        weights = LRPPropFunctions.detach_if_grad(grad_fn._saved_mat2) # d o

        #### TODO: (Aug 21, 2025) There is an overlooked case of when one of x, weights is None because one of them does not have requires_grad=True
        # However, this is most likely mitigated if the model is in training mode / using requires_grad as a whole.
        # Priority right now is getting the deterministic execution plan run mode working, can circle back to this later
        # because it will likely require some reworking on Promises as well. See the note I left in promise.py.

        #### TODO: Add an ablation for Attn-LRP vs traditional LRP

        z = x @ weights # s o

        if isinstance(r, Promise):
            r.setarg(z, grad_fn, lambda fcn: LRPPropFunctions.detach_if_grad(fcn._saved_self) @ LRPPropFunctions.detach_if_grad(fcn._saved_mat2))
            if r.complete:
                r = r.rin
            else:
                # If this is the first branch of the promise
                return r # We will check if the promise is complete in the graph traversal
        
        filter_val = grad_fn.metadata["relevance_filter"]

        if grad_fn.metadata["use_gamma"]:
            gamma = grad_fn.metadata["gamma"]
            return gamma_lrp_general(torch.matmul, None, {}, x, weights, z, r, gamma, filter_val)[:2]

        if grad_fn.metadata["use_z_plus"]:
            weights = weights.clamp(min=0.0)
            z = x @ weights

        rin_input, rin_weight = epsilon_lrp_matmul(x, weights, z, r, bilinear=grad_fn.metadata["use_bilinear"])

        # propagate relevance in parallel for input and weight, as specified by A.3.2 (Bilinear matmul) of AttnLRP: https://arxiv.org/pdf/2402.05602
        return relevance_filter(rin_input, filter_val), rin_weight

    @staticmethod
    @output_relevances
    @add_node_to_promise_path
    def BmmBackwardProp(grad_fn, r):

        if "pre_promise" in grad_fn.metadata and len((pre_promise := grad_fn.metadata["pre_promise"]).fwd_list) == 0:
            assert pre_promise.ready, f"Traversed Node after already placing a Pre-Promise, but the Promise is not ready, at {grad_fn}, topo-ind {grad_fn.metadata['topo_ind']}"
            z = pre_promise.arg
        else:
            mat1 = LRPPropFunctions.detach_if_grad(grad_fn._saved_self) # b s d
            mat2 = LRPPropFunctions.detach_if_grad(grad_fn._saved_mat2) # b d o

            z = torch.bmm(mat1, mat2) # b s o

            # assert r.shape == z.shape, f"r shape {r.shape} must match z shape {z.shape}"

            if isinstance(r, Promise):
                r.setarg(z, grad_fn, lambda fcn: LRPPropFunctions.detach_if_grad(fcn._saved_self) @ LRPPropFunctions.detach_if_grad(fcn._saved_mat2))
                if r.complete:
                    r = r.rin
                else:
                    # If this is the first branch of the promise
                    return r # We will check if the promise is complete in the graph traversal

        rin_mat1, rin_mat2 = epsilon_lrp_matmul(mat1, mat2, z, r, bilinear=grad_fn.metadata["use_bilinear"])

        # propagate relevance in parallel for mat1 and mat2
        return rin_mat1, rin_mat2
    
    @staticmethod
    @output_relevances
    @add_node_to_promise_path
    def DecomposedConvolutionBackwardProp(grad_fn, r):

        def conv_out(node):
            num_dims = len(node._saved_stride)

            if num_dims == 1:
                conv_f = F.conv1d
            elif num_dims == 2:
                conv_f = F.conv2d
            elif num_dims == 3:
                conv_f = F.conv3d
            else:
                raise ValueError(f"Expected {grad_fn} at topo-ind {grad_fn.metadata['topo_ind']} to be either 1, 2, or 3 dim, but {num_dims} was found.")

            return conv_f(LRPPropFunctions.detach_if_grad(node._saved_input), LRPPropFunctions.detach_if_grad(node._saved_weight), None, node._saved_stride, node._saved_padding, node._saved_dilation, node._saved_groups), conv_f

        # if "pre_promise" in grad_fn.metadata and len((pre_promise := grad_fn.metadata["pre_promise"]).fwd_list) == 0:
        #     assert pre_promise.ready, f"Traversed Node after already placing a Pre-Promise, but the Promise is not ready, at {grad_fn}, topo-ind {grad_fn.metadata['topo_ind']}"
        #     z = pre_promise.arg
        # else:
        z, conv_f = conv_out(grad_fn)

        if isinstance(r, Promise):
            r.setarg(z, grad_fn, lambda node: conv_out(node)[0])
            if r.complete:
                r = r.rin
            else:
                # If this is the first branch of the promise
                return r # We will check if the promise is complete in the graph traversal

        num_dims = len(grad_fn._saved_stride)
        if num_dims == 1:
            conv_T = F.conv_transpose1d
        elif num_dims == 2:
            conv_T = F.conv_transpose2d
        else:
            conv_T = F.conv_transpose3d

        x = LRPPropFunctions.detach_if_grad(grad_fn._saved_input)
        weights = LRPPropFunctions.detach_if_grad(grad_fn._saved_weight)

        filter_val = grad_fn.metadata["relevance_filter"]

        if grad_fn.metadata["use_gamma"]:
            gamma = grad_fn.metadata["gamma"]
            return gamma_lrp_general(conv_f, conv_T, {"stride": grad_fn._saved_stride, "padding": grad_fn._saved_padding, "dilation": grad_fn._saved_dilation, "groups": grad_fn._saved_groups}, x, weights, z, r, gamma, filter_val)

        if grad_fn.metadata["use_z_plus"]:
            weights = weights.clamp(min=0.0)
        
        sign = ((z == 0.).to(z) + z.sign())
        s = r / (z + sign * epsilon)
        c = conv_T(s, weights, None, grad_fn._saved_stride, grad_fn._saved_padding, 0, grad_fn._saved_groups, grad_fn._saved_dilation)

        if c.shape != x.shape:
            # Assume c is larger due to padding, center-crop down to input H, W
            _, _, H, W = x.shape
            c = c[:, :, :H, :W]

        r_input = x * c
        r_input = r_input

        grad_w = torch.nn.grad.conv2d_weight(x, weights.shape, s, grad_fn._saved_stride, grad_fn._saved_padding, grad_fn._saved_dilation, grad_fn._saved_groups)
        r_weight = weights * grad_w  # elementwise, scales by weight itself
        r_weight = r_weight

        return relevance_filter(r_input, filter_val), r_weight, 0.0
    
    @staticmethod
    @output_relevances
    @add_node_to_promise_path
    def MaxPool2DWithIndicesBackwardProp(grad_fn, r):
        if isinstance(r, Promise):
            r.setarg(grad_fn._saved_result1, grad_fn, "_saved_result1")
            if r.complete:
                r = r.rin
            else:
                return r
        return relevance_filter(grad_fn(r))
    
    @staticmethod
    @output_relevances
    @add_node_to_promise_path
    def MaxPool3DWithIndicesBackwardProp(grad_fn, r):
        if isinstance(r, Promise):
            r.setarg(grad_fn._saved_result1, grad_fn, "_saved_result1")
            if r.complete:
                r = r.rin
            else:
                return r
        return relevance_filter(grad_fn(r))
    
    @staticmethod
    @output_relevances
    @add_node_to_promise_path
    def AdaptiveAvgPool2DBackwardProp(grad_fn, r):
        if isinstance(r, Promise):
            def apply_adaptive_avg_pool(fcn):
                pool_shape = tuple(list(fcn._input_metadata[0].shape)[-2:])
                return F.adaptive_avg_pool2d(fcn._saved_self, pool_shape)
            r.setarg(apply_adaptive_avg_pool(grad_fn), grad_fn, apply_adaptive_avg_pool)
            if r.complete:
                r = r.rin
            else:
                return r

        # ChatGPT'd this, TODO: come back and understand the math
        x = grad_fn._saved_self
        N, C, H_in, W_in = x.shape
        H_out, W_out = tuple(list(grad_fn._input_metadata[0].shape)[-2:])

        rin = torch.zeros_like(x)

        for oh in range(H_out):
            h_start = int(torch.floor(torch.tensor(oh * H_in / H_out)))
            h_end   = int(torch.ceil(torch.tensor((oh + 1) * H_in / H_out)))
            for ow in range(W_out):
                w_start = int(torch.floor(torch.tensor(ow * W_in / W_out)))
                w_end   = int(torch.ceil(torch.tensor((ow + 1) * W_in / W_out)))

                region_size = (h_end - h_start) * (w_end - w_start)

                # Distribute R[:, :, oh, ow] equally over the region
                rin[:, :, h_start:h_end, w_start:w_end] += (
                    r[:, :, oh:oh+1, ow:ow+1] / region_size
                )

        return relevance_filter(rin)
    
    @staticmethod
    @output_relevances
    @add_node_to_promise_path
    def ScaledDotProductEfficientAttentionBackwardProp(grad_fn, r):
        if isinstance(r, Promise):
            r.setarg(grad_fn._saved_output)
            if r.complete:
                r = r.rin
            else:
                return r

        is_cpu = "is_cpu" in grad_fn.metadata
        is_causal = grad_fn._saved_is_causal

        attn_mask : torch.Tensor = None if (mask := getattr(grad_fn, "_saved_attn_mask", None)) is None else LRPPropFunctions.detach_if_grad(mask)
        attn_bias : torch.Tensor = None if (bias := getattr(grad_fn, "_saved_attn_bias", None)) is None else LRPPropFunctions.detach_if_grad(bias)
        query : torch.Tensor = LRPPropFunctions.detach_if_grad(grad_fn._saved_query)
        key : torch.Tensor = LRPPropFunctions.detach_if_grad(grad_fn._saved_key)
        value : torch.Tensor = LRPPropFunctions.detach_if_grad(grad_fn._saved_value)
        logsumexp : torch.Tensor = LRPPropFunctions.detach_if_grad(grad_fn._saved_logsumexp) if is_cpu else LRPPropFunctions.detach_if_grad(grad_fn._saved_log_sumexp) # Annoying difference between CPU and GPU implementations
        output : torch.Tensor = LRPPropFunctions.detach_if_grad(grad_fn._saved_output)
        scale : float = grad_fn._saved_scale if grad_fn._saved_scale is not None else 1 / math.sqrt(query.size(-1))

        if hasattr(grad_fn, "_saved_enable_gqa"):
            enable_gqa = grad_fn._saved_enable_gqa
        else:
            enable_gqa = enable_gqa = query.shape[:-2] != key.shape[:-2] or query.shape[:-2] != value.shape[:-2]

        k_zeros = torch.zeros_like(key)

        if enable_gqa:
            # We need to do the repeat anyway to get the values for propagating relevance,
            # so keep the grad_fns to collapse the relevances back down if GQA is enabled.
            # Note that x.repeat_interleave(repeats, dim) = x.unsqueeze(dim + 1).expand(*x.shape[:dim + 1], repeats, *x.shape[dim + 1:]).clone().view((*x.shape[:dim], x.shape[dim] * repeats, *x.shape[dim:]))
            # And the grad_fns of these intermediate steps are what gets returned by x.repeat_interleave(repeats, dim).grad_fn.
            # Since they are all shape/copy ops, we can reuse for k and v if they are the same latent size.
            key.requires_grad_()
            value.requires_grad_()
            key = key.repeat_interleave(query.size(-3)//key.size(-3), -3)
            value = value.repeat_interleave(query.size(-3)//value.size(-3), -3)
            repeat_interleave_grad_fn_k = key.grad_fn
            repeat_interleave_grad_fn_v = key.grad_fn
            if value.size(-3) != key.size(-3):
                repeat_interleave_grad_fn_v = value.grad_fn
            key.requires_grad_(False)
            value.requires_grad_(False)
            key = LRPPropFunctions.detach_if_grad(key)
            value = LRPPropFunctions.detach_if_grad(value)
            

        L, S = query.size(-2), key.size(-2)

        # Remake bias
        if attn_bias is None:
            attn_bias = torch.zeros(L, S, dtype=query.dtype, device=query.device, requires_grad=LRPPropFunctions.with_grad)
            if is_causal:
                # Causal bias gets computed and added to attn_bias
                temp_mask = torch.ones(L, S, dtype=torch.bool, device=query.device).tril(diagonal=0)
                attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))

        if attn_mask is not None:
            # User-provided mask gets added to attn_bias
            if attn_mask.dtype == torch.bool:
                attn_bias.masked_fill_(attn_mask.logical_not(), float("-inf"))
            else:
                attn_bias = attn_mask + attn_bias

        # Have to do some forward pass for propagation :(
        unscaled_attn_weights = query @ key.transpose(-2, -1)
        unbiased_attn_weights = unscaled_attn_weights * scale
        attn_weights = unbiased_attn_weights + attn_bias

        # Compute softmax from logsumexp
        if PromiseBucket.dtype != torch.float32:
            if logsumexp.size(-1) > attn_weights.size(-1):
                # GPU implementation might use fixed size for tiling
                attn_logits = torch.exp(attn_weights - logsumexp[:,:,:attn_weights.size(-2)].unsqueeze(-1)).to(dtype=attn_weights.dtype)
            else:
                attn_logits = torch.exp(attn_weights - logsumexp.unsqueeze(-1)).to(dtype=attn_weights.dtype)
        else:
            attn_logits = attn_weights.softmax(dim=-1)
        del attn_bias

        # Distribute last matmul
        r_attn_logits, r_v = epsilon_lrp_matmul(attn_logits, value, output, r, bilinear=True)
        if enable_gqa:
            # Collapse r_v
            r_v = repeat_interleave_grad_fn_v(r_v)

        if not grad_fn.metadata["use_attn_lrp"]:
            if is_cpu:
                return torch.zeros_like(query), k_zeros, r_v
            else:
                return torch.zeros_like(query), k_zeros, r_v, 0.0

        # Distribute softmax
        r_attn_weights = unbiased_attn_weights.tril() * (r_attn_logits - attn_logits * r_attn_logits.sum(dim=-1, keepdim=True))
        del r_attn_logits

        # Distribute QK^T
        r_q, r_k = epsilon_lrp_matmul(query, key, unscaled_attn_weights, r_attn_weights, w_transposed=False, bilinear=True)
        del attn_weights, r_attn_weights

        if enable_gqa:
            # Collapse r_k
            r_k = repeat_interleave_grad_fn_k(r_k)
        
        if is_cpu:
            return r_q, r_k, r_v
        else:
            return r_q, r_k, r_v, 0.0
    
    @classmethod
    @output_relevances
    @add_node_to_promise_path
    def ScaledDotProductFlashAttentionForCpuBackwardProp(cls, grad_fn, r):
        grad_fn.metadata["is_cpu"] = True
        return cls.ScaledDotProductEfficientAttentionBackwardProp(grad_fn, r)

    @staticmethod
    @output_relevances
    @add_node_to_promise_path
    def NativeLayerNormBackwardProp(grad_fn, r):

        def layerNorm(fcn):
            x = LRPPropFunctions.detach_if_grad(fcn._saved_input)
            mean = LRPPropFunctions.detach_if_grad(fcn._saved_result1)
            gamma = LRPPropFunctions.detach_if_grad(fcn._saved_weight)
            beta = LRPPropFunctions.detach_if_grad(fcn._saved_bias)
            rec_stddev = LRPPropFunctions.detach_if_grad(fcn._saved_result2)
            normalized = (x - mean) * rec_stddev
            return normalized * gamma + beta

        if isinstance(r, Promise):
            r.setarg(layerNorm(grad_fn), grad_fn, layerNorm)
            if r.complete:
                r = r.rin
            else:
                # If this is the first branch of the promise
                return r # We will check if the promise is complete in the graph traversal

        # next_functions will correspond to input, weights, bias
        # We only care about propagating through the input for LayerNorm.
        # We can try some denoising here
        return r, 0.0, 0.0
    
    @staticmethod
    @output_relevances
    @add_node_to_promise_path
    def NativeBatchNormBackwardProp(grad_fn, r):

        def batchNorm(fcn):
            x = LRPPropFunctions.detach_if_grad(fcn._saved_input)
            mean = LRPPropFunctions.detach_if_grad(fcn._saved_result1)
            rec_stddev = LRPPropFunctions.detach_if_grad(fcn._saved_result2)
            normalized = (x - mean) * rec_stddev
            return normalized

        if isinstance(r, Promise):
            r.setarg(batchNorm(grad_fn), grad_fn, batchNorm)
            if r.complete:
                r = r.rin
            else:
                # If this is the first branch of the promise
                return r # We will check if the promise is complete in the graph traversal

        return r, 0.0, 0.0

    
    @staticmethod
    @output_relevances
    @add_node_to_promise_path
    def GeluBackwardProp(grad_fn, r):
        if isinstance(r, Promise):
            r.setarg(F.gelu(LRPPropFunctions.detach_if_grad(grad_fn._saved_self)), grad_fn, lambda fcn: torch.nn.GELU(LRPPropFunctions.detach_if_grad(fcn._saved_self)))
            if r.complete:
                r = r.rin
        return r
    
    @staticmethod
    @output_relevances
    @add_node_to_promise_path
    def SiluBackwardProp(grad_fn, r):
        if isinstance(r, Promise):
            def apply_silu(node):
                return F.silu(node._saved_self)
            r.setarg(apply_silu(grad_fn), grad_fn, apply_silu)
            if r.complete:
                r = r.rin
        return r
    
    @staticmethod
    @output_relevances
    @add_node_to_promise_path
    @skip_redundant_promise
    def SoftmaxBackwardProp(grad_fn, r):
        if not "use_attn_lrp" in grad_fn.metadata:
            result = grad_fn._saved_result.detach()
            if isinstance(r, Promise):
                r.setarg(result, grad_fn, "_saved_result")
                if r.complete:
                    r = r.rin
            return r

        promise = {
            "rout": r,
            "args": [None],
            "rins": [None],
            "ready": False,
            "complete": False,
            "parents": [],
        }
        if isinstance(r, Promise):
            promise["parents"] = [r]
            promise["rout"] = torch.zeros(r.fwd_shape, device=r.rout.device, dtype=r.rout.dtype, requires_grad=LRPPropFunctions.with_grad) # Placeholder for shape
        
        traversal_ind = grad_fn.metadata["topo_ind"]
        bucket = grad_fn.metadata["bucket"]

        promise = SoftmaxBackwardPromise(promise, traversal_ind, bucket, getattr(grad_fn, "_saved_result"), getattr(grad_fn, "_saved_dim"))

        if isinstance(r, Promise):
            r.arg_node_ind = traversal_ind
            r.children.append(promise)

        grad_fn.metadata["promise"] = promise

        return promise
    
    @classmethod
    @output_relevances
    @add_node_to_promise_path
    def SafeSoftmaxBackwardProp(cls, grad_fn, r):
        return cls.SoftmaxBackwardProp(grad_fn, r)
    
    @staticmethod
    @output_relevances
    @add_node_to_promise_path
    def PowBackwardProp(grad_fn, r):
        if isinstance(r, Promise):
            def get_result(node):
                return node._saved_self.pow(node._saved_exponent)
            r.setarg(get_result(grad_fn), grad_fn, get_result)
            if r.complete:
                r = r.rin
        return r
    
    @staticmethod
    @output_relevances
    @add_node_to_promise_path
    def SqrtBackwardProp(grad_fn, r):
        if isinstance(r, Promise):
            r.setarg(LRPPropFunctions.detach_if_grad(grad_fn._saved_result), grad_fn, "_saved_result")
            if r.complete:
                r = r.rin
        return r
    
    @staticmethod
    @output_relevances
    @add_node_to_promise_path
    def RsqrtBackwardProp(grad_fn, r):
        if isinstance(r, Promise):
            r.setarg(LRPPropFunctions.detach_if_grad(grad_fn._saved_result), grad_fn, "_saved_result")
            if r.complete:
                r = r.rin
        return r
    
    @staticmethod
    @output_relevances
    @add_node_to_promise_path
    def IdentityProp(grad_fn, r):
        """Placeholder for any missed operations, or general use for identity-rule operations."""
        num_rel_outs = len(grad_fn.next_functions)

        if isinstance(r, Promise):
            if hasattr(grad_fn, "_saved_result"):
                r.setarg(LRPPropFunctions.detach_if_grad(grad_fn._saved_result), grad_fn, "_saved_result")
                if r.complete:
                    return r.rin
                return r
            else:
                raise ValueError(f"{grad_fn} promise handling is currently unsupported. Please open an Issue or PR for implementing the propagation function.")

        if num_rel_outs == 1:
            return r
        return tuple([ r / num_rel_outs for _ in range(num_rel_outs) ])
    
    @staticmethod
    @output_relevances
    @add_node_to_promise_path
    def IndexPutFirstAxisBackwardProp(grad_fn, r):
        """Identity but needs custom output"""
        return r, 0.0
    
    @staticmethod
    @output_relevances
    @add_node_to_promise_path
    def IndexFirstAxisBackwardProp(grad_fn, r):
        """Identity but needs custom output"""
        return r, 0.0
    
    @staticmethod
    @output_relevances
    @add_node_to_promise_path
    def AccumulateGradProp(grad_fn, r):
        if isinstance(r, Promise):
            r.setarg(LRPPropFunctions.detach_if_grad(grad_fn.variable), grad_fn, "variable")
        return 0.0

    @staticmethod
    @output_relevances
    @add_node_to_promise_path
    def LRPCheckpointBackwardProp(grad_fn, r):
        saved_input = LRPPropFunctions.detach_if_grad(grad_fn.saved_tensors[0])
        if isinstance(r, Promise):
            r.setarg(saved_input, grad_fn, lambda fcn: LRPPropFunctions.detach_if_grad(fcn.saved_tensors[0]))
            if r.complete:
                r = r.rin
                LRPCheckpoint.save_val(grad_fn, r)
            else:
                return r
        else:
            LRPCheckpoint.save_val(grad_fn, r)
        return r
    
    @staticmethod
    @output_relevances
    @add_node_to_promise_path
    def EmbeddingBackwardProp(grad_fn, r):
        idxs = grad_fn._saved_indices
        if isinstance(r, Promise):
            r.setarg(grad_fn.next_functions[0][0].variable[idxs], grad_fn, lambda fcn: fcn.next_functions[0][0].variable[fcn._saved_indices])
            if r.complete:
                r = r.rin
            else:
                return r
        r_input = r.sum(dim=-1)
        grad_fn.metadata["relevance"] = r_input
        return r_input

    @classmethod
    def generate_prop_fcn_map(cls, names: list[str]):
        """Creates a dict to map grad_fn names to their corresponding LRP propagation
        functions"""
        fcn_map = {}
        for name in names:
            fcn_name = name + "Prop"
            if name[-1].isdigit():
                fcn_name = name[:-1] + "Prop"
            if (hasattr(cls, fcn_name) and callable(fcn := getattr(cls, fcn_name))
                and not fcn_name.startswith("_")):
                fcn_map[name] = fcn
            else:
                fcn_map[name] = cls.IdentityProp
        return fcn_map
