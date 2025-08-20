import torch
import torch.nn.functional as F
from add_backward_promise import AddBackwardPromise
from sum_backward_promise import SumBackwardPromise
from promise import Promise
from util import (
    epsilon,
    renormalize_epsilon,
    renormalize_epsilon_scalar,
    shift_and_renormalize,
    DEBUG,
    LRPCheckpoint,
)

"""
For all these functions, grad_fn is the autograd Node returned from traversing the autograd graph.
r is the relevance tensor of the output of the given Node.
"""


def output_relevances(func):
    if not DEBUG:
        return func
    def wrapper(*args, **kwargs):
        res = func(*args, **kwargs)
        grad_fn = args[1]
        r = args[2]
        if type(grad_fn).__name__ == "AddBackward0":
            print(f"{grad_fn}:", end="") 
            if isinstance(r, Promise):
                print(r.rout.nansum())
            else:
                print(r.nansum())
        if (not isinstance(r, Promise)) and \
                not isinstance(res, Promise) \
                and (not isinstance(res, tuple) or not isinstance(res[0], Promise)):
            print(f"{grad_fn}: ", end="")
            rout = r.nansum()
            rins = None
            if isinstance(res, tuple):
                rins = ((res[0].nansum() if isinstance(res[0], torch.Tensor) else res[0]) +
                        (res[1].nansum() if isinstance(res[1], torch.Tensor) else res[1]))
            else:
                rins = res.sum()
            print(rout, rins)
        return res
    return wrapper

def add_node_to_promise_path(func):
    def wrapper(*args, **kwargs):
        r = args[2]
        grad_fn = args[1]
        if isinstance(r, Promise):
            r.add_to_path(grad_fn.metadata["topo_ind"])
        return func(*args, **kwargs)
    return wrapper
class LRPPropFunctions:

    @classmethod
    @output_relevances
    @add_node_to_promise_path
    def AddBackwardProp(cls, grad_fn, r):
        # IMPORTANT: AddBackward0 does not actually store any operands of the addition, so we have
        # to get a bit tricky.
        # The idea is to return a "promise", a dict wrapped in a class which contains the outgoing relevance, and
        # placeholders for the operands and their respective relevances.
        # From what I know right now, AddBackward0 is the only math-op grad_fn that does this, so the hope is
        # that we pass this promise down the graph, and we encounter one of:
        # 1. AccumulateGrad or another math-op that we can get the result from
        # 2. A function that follows the identity or uniform rule like GeluBackward0 or LayerNormBackward0
        # 3. A mutation function like SliceBackward0 or ReshapeBackward0
        # 4. (worst case) Another AddBackward0
        # For case 2 and 3, we would have to keep an arbitrarily composable function which progressively
        # nests the operations that must be done on the result, once it is found, to make it equivalent
        # to the downstream addition operand. When we find a node with the result, we simply apply f(result)
        # to get the actual operand for the original addition.
        # However, we will also need to keep a similar function but for going backwards from the addition
        # back to the result node, but this time for the relevance.
        # If at this time, both operands have been found, compute and store the relevances for both in the
        # promise. If not, move this node to the back of the queue.
        # For case 4, we would simply have to nest a promise within the existing promise.
        # So the only time this algorithm will branch is if there are multiple additions with no result-
        # yielding grad_fn's in between.

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
            promise["rout"] = torch.zeros_like(r.rout) # Placeholder for shape

        traversal_ind = grad_fn.metadata["topo_ind"]

        promise1 = AddBackwardPromise(promise, traversal_ind, 0)
        promise2 = AddBackwardPromise(promise, traversal_ind, 1)

        promise1.other_branch = promise2
        promise2.other_branch = promise1

        if isinstance(r, Promise):
            r.children = [(promise1, promise2)]

        grad_fn.metadata["promise"] = promise

        return promise1, promise2
    
    @classmethod
    @output_relevances
    @add_node_to_promise_path
    def SumBackwardProp(cls, grad_fn, r):
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
            promise["rout"] = torch.zeros_like(r.rout) # Placeholder for shape
        
        traversal_ind = grad_fn.metadata["topo_ind"]

        promise = SumBackwardPromise(promise, traversal_ind, getattr(grad_fn, "_saved_dim"), getattr(grad_fn, "_saved_keepdim"))

        if isinstance(r, Promise):
            r.children = [promise]

        grad_fn.metadata["promise"] = promise

        return promise
    
    @classmethod
    @output_relevances
    @add_node_to_promise_path
    def MeanBackwardProp(cls, grad_fn, r):
        """Mean is just a scaled sum by 1/n, so the ratios of all elements
        and their contributions should still be the same as if they were a normal
        sum."""
        return cls.SumBackwardProp(grad_fn, r)

    @classmethod
    @output_relevances
    @add_node_to_promise_path
    def MaxBackwardProp(cls, grad_fn, r):
        max_type = 0 if hasattr(grad_fn, "_saved_dim") else 1

        if isinstance(r, Promise):
            if max_type == 0:
                r.setarg(grad_fn._saved_result)
                if r.complete:
                    r = r.rin
                else:
                    return r
            else:
                r.nest_fwd(lambda x: torch.max(x, dim=grad_fn._saved_dim, keepdim=grad_fn._saved_keepdim))
                r.nest_bwd(grad_fn)
                return r

        return grad_fn(r) # Reuse autograd implementation

    @classmethod
    @output_relevances
    @add_node_to_promise_path
    def ViewBackwardProp(cls, grad_fn, r):
        upstream_shape = grad_fn._saved_self_sym_sizes
        if isinstance(r, Promise):
            target_shape = r.fwd_shape
            r.nest_fwd(lambda x: x.reshape(target_shape))
            r.nest_bwd(lambda x: x.reshape(upstream_shape))
            r.fwd_shape = upstream_shape
            return r
        return r.reshape(upstream_shape)

    @classmethod
    @output_relevances
    @add_node_to_promise_path
    def UnsafeViewBackwardProp(cls, grad_fn, r):
        upstream_shape = grad_fn._saved_self_sym_sizes
        if isinstance(r, Promise):
            target_shape = r.fwd_shape
            r.nest_fwd(lambda x: x.reshape(target_shape))
            r.nest_bwd(lambda x: x.reshape(upstream_shape))
            r.fwd_shape = upstream_shape
            return r
        return r.reshape(upstream_shape)

    @classmethod
    @output_relevances
    @add_node_to_promise_path
    def ReshapeBackwardProp(cls, grad_fn, r):
        upstream_shape = grad_fn._saved_self_sym_sizes
        if isinstance(r, Promise):
            target_shape = r.fwd_shape
            r.nest_fwd(lambda x: torch.reshape(x, target_shape))
            r.nest_bwd(lambda x: torch.reshape(x, upstream_shape))
            r.fwd_shape = upstream_shape
            return r
        return torch.reshape(r, upstream_shape)

    @classmethod
    @output_relevances
    @add_node_to_promise_path
    def ReshapeAliasBackwardProp(cls, grad_fn, r):
        upstream_shape = grad_fn._saved_self_sym_sizes
        if isinstance(r, Promise):
            target_shape = r.fwd_shape
            r.nest_fwd(lambda x: torch.reshape(x, target_shape))
            r.nest_bwd(lambda x: torch.reshape(x, upstream_shape))
            r.fwd_shape = upstream_shape
            return r
        return torch.reshape(r, upstream_shape)

    @classmethod
    @output_relevances
    @add_node_to_promise_path
    def SliceBackwardProp(cls, grad_fn, r):
        # Assumes the index corresponding to _saved_start in the forward pass is non-negative.
        # If it was negative-indexed, i.e. x[-i:] autograd saves the index as INT_MAX - i.
        upstream_shape = grad_fn._saved_self_sym_sizes
        sliced_dim = grad_fn._saved_dim
        start = grad_fn._saved_start # TODO: Come back to take care of the negative index case.
        full_size = upstream_shape[sliced_dim]
        end = start + r.shape[sliced_dim]

        # We wish to pad r so that it becomes the correct size along the sliced dimension
        pad = []
        to_pad = [start, full_size - end]

        # Iterate in reverse order, since F.pad() takes in dims from last to first,
        # see https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.pad.html
        # All dims other than sliced_dim should be 0, 0
        for dim in range(len(upstream_shape) - 1, -1, -1):
            pad += [0, 0] if dim != sliced_dim else to_pad
        pad = tuple(pad)

        if isinstance(r, Promise):
            r.nest_fwd(lambda x: torch.ops.aten.slice(x, sliced_dim, start, end))
            r.nest_bwd(lambda x: F.pad(x, pad))
            r.fwd_shape = upstream_shape
            return r
        return F.pad(r, pad)

    @classmethod
    @output_relevances
    @add_node_to_promise_path
    def IndexBackwardProp(cls, grad_fn, r):
        # An Index can be compound, unlike Slice, i.e. a[[0,1], [1,2]] is ONE Index op, whereas a[:,1:] is TWO Slice ops.
        # This is because (in this case) the second Slice depends on the first. It's saying that from the result of
        # the first slice, for each element, select index 1 from the first, and index 2 from the second (assuming of
        # course that 1 and 2 are in bounds for each element returned by the first slice).
        # Therefore, the length of the first Slice acts as an upper bound for the length of the Slices that succeed it.
        # If you wanted it to select indices 1 and 2 for each resulting element, you would just use a Slice for the last
        # dim instead of an Index.
        upstream_shape = grad_fn._saved_self_sym_sizes
        
        idxs = [ torch.tensor(x) if x is not None else None for x in grad_fn._saved_indices ]

        def undoIndex(x):
            out = torch.zeros(upstream_shape, dtype=x.dtype, device=x.device)
            return torch.ops.aten.index_put(out, idxs, x)

        if isinstance(r, Promise):
            r.nest_fwd(lambda x: torch.ops.aten.index(x, idxs))
            r.nest_bwd(undoIndex)
            r.fwd_shape = upstream_shape
            return r
        return undoIndex(r)

    @classmethod
    @output_relevances
    @add_node_to_promise_path
    def SelectBackwardProp(cls, grad_fn, r):
        upstream_shape = grad_fn._saved_self_sym_sizes
        dim = grad_fn._saved_dim
        idx = grad_fn._saved_index

        def undoSelect(x):
            out = torch.zeros(upstream_shape, dtype=x.dtype, device=x.device)
            out.select(dim, idx).copy_(x)
            return out

        if isinstance(r, Promise):
            r.nest_fwd(lambda x: torch.select(x, dim, idx))
            r.nest_bwd(undoSelect)
            r.fwd_shape = upstream_shape
            return r

        return undoSelect(r)

    @classmethod
    @output_relevances
    @add_node_to_promise_path
    def TBackwardProp(cls, grad_fn, r):
        # Not sure why TBackward is different from TransposeBackward, but it seems like this is only
        # in Linear layer matmuls on W for xW^T before Mm and Addmm operations.
        assert len(r.shape) == 2, "Assumption was that matrix would be 2d Linear weights." # For now assume that it is only 2d matmuls for Linear layers.

        transpose = lambda x: x.T

        if isinstance(r, Promise):
            r.nest_fwd(transpose)
            r.nest_bwd(transpose)
            new_shape = list(r.fwd_shape)
            new_shape[0], new_shape[1] = new_shape[1], new_shape[0]
            r.fwd_shape = tuple(new_shape)
            return r

        return transpose(r)

    @classmethod
    @output_relevances
    @add_node_to_promise_path
    def TransposeBackwardProp(cls, grad_fn, r):
        dim1 = grad_fn._saved_dim0
        dim2 = grad_fn._saved_dim1

        if dim1 == 2**32 - 2:
            dim1 = -2
        if dim2 == 2**32 - 1:
            dim2 = -1

        swapaxes = lambda x: torch.swapaxes(x, dim1, dim2)

        if isinstance(r, Promise):
            r.nest_fwd(swapaxes)
            r.nest_bwd(swapaxes)
            new_shape = list(r.fwd_shape)
            new_shape[dim1], new_shape[dim2] = new_shape[dim2], new_shape[dim1]
            r.fwd_shape = tuple(new_shape)
            return r
        
        return swapaxes(r)

    @classmethod
    @output_relevances
    @add_node_to_promise_path
    def PermuteBackwardProp(cls, grad_fn, r):
        dims = grad_fn._saved_dims

        undo_dims = [0] * len(dims)
        for old_dim, new_dim in enumerate(dims):
            undo_dims[new_dim] = old_dim

        upstream_shape = [ r.shape[i] for i in undo_dims ]

        if isinstance(r, Promise):
            r.nest_fwd(lambda x: x.permute(dims))
            r.nest_bwd(lambda x: x.permute(undo_dims).clone())
            r.fwd_shape = upstream_shape
            return r
        return r.permute(undo_dims).clone()

    @classmethod
    @output_relevances
    @add_node_to_promise_path
    def ExpandBackwardProp(cls, grad_fn, r):
        upstream_shape = grad_fn._saved_self_sym_sizes
        downstream_shape = r.shape
        assert len(upstream_shape) == len(downstream_shape), "Expand should not increase number of dimensions."

        expand_input = [ dim2 if dim1 != dim2 else -1 for dim1, dim2 in zip(upstream_shape, downstream_shape) ]

        def undoExpand(x):
            for i, expand_dim in enumerate(expand_input):
                if expand_dim != -1:
                    x = x.select(i, 0).unsqueeze(i) * (upstream_shape[i] / downstream_shape[i])
            return x

        expand = lambda x: x.expand(*expand_input)

        if isinstance(r, Promise):
            r.nest_fwd(expand)
            r.nest_bwd(undoExpand)
            r.fwd_shape = upstream_shape
            return r

        return undoExpand(r)

    @classmethod
    @output_relevances
    @add_node_to_promise_path
    def MulBackwardProp(cls, grad_fn, r):
        arg1 = grad_fn._saved_self
        arg2 = grad_fn._saved_other

        if isinstance(r, Promise):
            if arg1 is None:
                r.nest_fwd(lambda x: x * arg2)
            else:
                r.setarg(arg1 * arg2, grad_fn)
                if r.complete:
                    r = r.rin
                else:
                    return r # We will check if the promise is complete in the graph traversal

        if arg1 is None:
            # Tensor-scalar product, disregard scalar
            return r, 0.0

        denom = arg1.abs() + arg2.abs() + epsilon
        r1 = (arg1.abs() / denom) * r
        r2 = (arg2.abs() / denom) * r

        r1, r2 = renormalize_epsilon(r, r1, r2)
        # print(f"MulBackward relevance sums: out {r.sum()}, in {(r1 + r2).sum()}")

        return r1, r2

    @classmethod
    @output_relevances
    @add_node_to_promise_path
    def DivBackwardProp(cls, grad_fn, r):
        arg1 = grad_fn._saved_self
        arg2 = grad_fn._saved_other

        if isinstance(r, Promise):
            if arg1 is None:
                r.nest_fwd(lambda x: x / arg2)
            else:
                r.setarg(arg1 / arg2, grad_fn)
                if r.complete:
                    r = r.rin
                else:
                    return r # We will check if the promise is complete in the graph traversal

        if arg1 is None:
            # Tensor-scalar product, disregard scalar
            return r, 0.0

        denom = arg1.abs() + (1 / arg2).abs() + epsilon
        r1 = (arg1.abs() / denom) * r
        r2 = ((1 / arg2).abs() / denom) * r

        r1, r2 = renormalize_epsilon(r, r1, r2)
        # print(f"DivBackward relevance sums: out {r.sum()}, in {(r1 + r2).sum()}")

        return r1, r2

    @classmethod
    @output_relevances
    @add_node_to_promise_path
    def MmBackwardProp(cls, grad_fn, r):
        x = grad_fn._saved_self # s d
        weights = grad_fn._saved_mat2 # d o
        z = x @ weights # s o
        z_stabilized = z + epsilon * z.sign()

        if isinstance(r, Promise):
            r.setarg(z, grad_fn)
            if r.complete:
                r = r.rin
            else:
                # If this is the first branch of the promise
                return r # We will check if the promise is complete in the graph traversal

        contribs = x.unsqueeze(2) * weights.unsqueeze(0) # s d o
        z_uns = z_stabilized.unsqueeze(1) # s d o
        ratio = contribs / z_uns
        r_in = r.unsqueeze(1) * ratio

        # Need to see if the other argument is just a weight or input-dependent
        next_node = grad_fn.next_functions[1][0]
        if next_node is not None and type(next_node).__name__ != "AccumulateGrad" \
            and type(next_node.next_functions[0][0]).__name__ != "AccumulateGrad":
            # split relevance between input and weight
            c1 = x.nansum() ** 2
            c2 = weights.nansum() ** 2
            denom = c1 + c2 + epsilon
            r1 = (c1 / denom) * r_in.nansum(dim=-1)
            r2 = (c2 / denom) * r_in.nansum(dim=0)
            return renormalize_epsilon_scalar(r_in, r1, r2)
        else:
            # propagate relevance in parallel for input and weight
            return shift_and_renormalize(r, r_in.sum(dim=-1)), torch.zeros_like(weights)

    @classmethod
    @output_relevances
    @add_node_to_promise_path
    def BmmBackwardProp(cls, grad_fn, r):
        mat1 = grad_fn._saved_self # b s d
        mat2 = grad_fn._saved_mat2 # b d o
        z = torch.bmm(mat1, mat2) # b s o

        assert r.shape == z.shape, f"r shape {r.shape} must match z shape {z.shape}"

        z_stabilized = z + epsilon * z.sign()

        if isinstance(r, Promise):
            r.setarg(z, grad_fn)
            if r.complete:
                r = r.rin
            else:
                # If this is the first branch of the promise
                return r # We will check if the promise is complete in the graph traversal

        contribs = mat1.unsqueeze(3) * mat2.unsqueeze(1) # b s d o
        z_uns = z_stabilized.unsqueeze(2) # b s d o
        ratio = contribs / z_uns
        r_in = ratio * r.unsqueeze(2)

        # Need to see if the other argument is just a weight or input-dependent
        next_node = grad_fn.next_functions[1][0]
        if next_node is not None and type(next_node).__name__ != "AccumulateGrad" \
            and type(next_node.next_functions[0][0]).__name__ != "AccumulateGrad":
            # split relevance between mat1 and mat2
            c1 = mat1.nansum() ** 2
            c2 = mat2.nansum() ** 2
            denom = c1 + c2 + epsilon
            r1 = (c1 / denom) * r_in.nansum(dim=3)
            r2 = (c2 / denom) * r_in.nansum(dim=1)
            return renormalize_epsilon_scalar(r_in, r1, r2)
        else:
            # propagate relevance in parallel for mat1 and mat2
            return shift_and_renormalize(r, r_in.nansum(dim=3)), torch.zeros_like(mat2)

    @classmethod
    @output_relevances
    @add_node_to_promise_path
    def NativeLayerNormBackwardProp(cls, grad_fn, r):
        input_ = grad_fn._saved_input
        mean = grad_fn._saved_result1
        gamma = grad_fn._saved_weight
        beta = grad_fn._saved_bias
        rec_stddev = grad_fn._saved_result2

        def layerNorm(x):
            normalized = (x - mean) * rec_stddev
            return normalized * gamma + beta

        if isinstance(r, Promise):
            r.setarg(layerNorm(input_), grad_fn)
            if r.complete:
                r = r.rin
            else:
                # If this is the first branch of the promise
                return r # We will check if the promise is complete in the graph traversal

        # next_functions will correspond to input, weights, bias
        # We only care about propagating through the input for LayerNorm.
        return r, 0.0, 0.0
    
    @classmethod
    @output_relevances
    @add_node_to_promise_path
    def GeluBackwardProp(cls, grad_fn, r):
        if isinstance(r, Promise):
            r.setarg(torch.nn.GELU(grad_fn._saved_self), grad_fn)
            if r.complete:
                r = r.rin
        return r
    
    @classmethod
    @output_relevances
    @add_node_to_promise_path
    def SoftmaxBackwardProp(cls, grad_fn, r):
        if isinstance(r, Promise):
            r.setarg(grad_fn._saved_result, grad_fn)
            if r.complete:
                r = r.rin
        return r

    @classmethod
    @output_relevances
    @add_node_to_promise_path
    def IdentityProp(cls, grad_fn, r):
        """Placeholder for any missed operations, or general use for identity-rule operations."""
        num_rel_outs = len(grad_fn.next_functions)

        if isinstance(r, AddBackwardPromise):
            if hasattr(grad_fn, "_saved_result"):
                r.setarg(grad_fn._saved_result)
                if r.complete:
                    return r.rin
                return r
            else:
                raise ValueError(f"{grad_fn} promise handling is currently unsupported. Please open an Issue or PR for implementing the propagation function.")

        if num_rel_outs == 1:
            return r
        return tuple([ r / num_rel_outs for _ in range(num_rel_outs) ])
    
    @classmethod
    @output_relevances
    @add_node_to_promise_path
    def IndexPutFirstAxisBackwardProp(cls, grad_fn, r):
        """Identity but needs custom output"""
        return r, 0.0
    
    @classmethod
    @output_relevances
    @add_node_to_promise_path
    def IndexFirstAxisBackwardProp(cls, grad_fn, r):
        """Identity but needs custom output"""
        return r, 0.0
    
    @classmethod
    @output_relevances
    @add_node_to_promise_path
    def AccumulateGradProp(cls, grad_fn, r):
        if isinstance(r, Promise):
            r.setarg(grad_fn.variable)
        return 0.0

    @classmethod
    @output_relevances
    @add_node_to_promise_path
    def LRPCheckpointBackwardProp(cls, grad_fn, r):
        saved_input = grad_fn._saved_tensors[0]
        if isinstance(r, Promise):
            r.setarg(saved_input)
            if r.complete:
                r = r.rin
            else:
                return r
        else:
            LRPCheckpoint.save_val(grad_fn, r)
        return r
    
    @classmethod
    @output_relevances
    @add_node_to_promise_path
    def EmbeddingBackwardProp(cls, grad_fn, r):
        idxs : torch.Tensor = grad_fn._saved_indices
        weights = grad_fn.next_functions[0][0].variable # BIG assumption here

        def undoEmbedding(x):
            out = torch.zeros_like(weights)
            if idxs.dim() == 1:
                out.index_add_(0, idxs, x)
            else:
                for inds in idxs:
                    out.index_add_(0, inds, x)
            return out

        if isinstance(r, Promise):
            target_shape = r.fwd_shape
            r.nest_fwd(lambda x: x[idxs.flatten()].reshape(target_shape))
            r.nest_bwd(undoEmbedding)
            r.fwd_shape = weights.shape
            return r

        return undoEmbedding(r)

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
