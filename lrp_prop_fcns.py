import torch
import torch.nn.functional as F
from add_backward_promise import AddBackwardPromise
from util import epsilon, renormalize_epsilon

"""
For all these functions, grad_fn is the autograd Node returned from traversing the autograd graph.
r is the relevance tensor of the output of the given Node.
"""

class LRPPropFunctions:

    @classmethod
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
            "parent": None,
        }
        if isinstance(r, AddBackwardPromise):
            promise["parent"] = r
            promise["rout"] = torch.zeros(r.fwd_shape) # Placeholder for shape

        promise1 = AddBackwardPromise(promise, 0)
        promise2 = AddBackwardPromise(promise, 1)

        promise1.other_branch = promise2
        promise2.other_branch = promise1

        if isinstance(r, AddBackwardPromise):
            r.children = [promise1, promise2]

        grad_fn.metadata["promise"] = promise

        return promise1, promise2

    @classmethod
    def ViewBackwardProp(cls, grad_fn, r):
        upstream_shape = grad_fn._saved_self_sym_sizes
        if isinstance(r, AddBackwardPromise):
            target_shape = r.fwd_shape
            r.nest_fwd(lambda x: torch.view(x, target_shape))
            r.nest_bwd(lambda x: torch.view(x, upstream_shape))
            r.fwd_shape = upstream_shape
            return r
        return torch.view(r, upstream_shape)

    @classmethod
    def UnsafeViewBackwardProp(cls, grad_fn, r):
        upstream_shape = grad_fn._saved_self_sym_sizes
        if isinstance(r, AddBackwardPromise):
            target_shape = r.fwd_shape
            r.nest_fwd(lambda x: torch.view(x, target_shape))
            r.nest_bwd(lambda x: torch.view(x, upstream_shape))
            r.fwd_shape = upstream_shape
            return r
        return torch.view(r, upstream_shape)

    @classmethod
    def ReshapeBackwardProp(cls, grad_fn, r):
        upstream_shape = grad_fn._saved_self_sym_sizes
        if isinstance(r, AddBackwardPromise):
            target_shape = r.fwd_shape
            r.nest_fwd(lambda x: torch.reshape(x, target_shape))
            r.nest_bwd(lambda x: torch.reshape(x, upstream_shape))
            r.fwd_shape = upstream_shape
            return r
        return torch.reshape(r, grad_fn._saved_self_sym_sizes)

    @classmethod
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

        if isinstance(r, AddBackwardPromise):
            r.nest_fwd(lambda x: torch.ops.aten.slice(x, sliced_dim, start, end))
            r.nest_bwd(lambda x: F.pad(x, pad))
            r.fwd_shape = upstream_shape
            return r
        return F.pad(r, pad)

    @classmethod
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

        if isinstance(r, AddBackwardPromise):
            r.nest_fwd(lambda x: torch.ops.aten.index(x, idxs))
            r.nest_bwd(undoIndex)
            r.fwd_shape = upstream_shape
            return r
        return undoIndex(r)

    @classmethod
    def SelectBackwardProp(cls, grad_fn, r):
        upstream_shape = grad_fn._saved_self_sym_sizes
        dim = grad_fn._saved_dim
        idx = grad_fn._saved_index

        def undoSelect(x):
            out = torch.zeros(upstream_shape, dtype=x.dtype, device=x.device)
            x_expanded = torch.unsqueeze(x, dim)
            out.select(dim, idx).copy_(x_expanded)
            return out

        if isinstance(r, AddBackwardPromise):
            r.nest_fwd(lambda x: torch.select(x, dim, idx))
            r.nest_bwd(undoSelect)
            r.fwd_shape = upstream_shape
            return r

        return undoSelect(r)

    @classmethod
    def TBackwardProp(cls, grad_fn, r):
        # Not sure why TBackward is different from TransposeBackward, but it seems like this is only
        # in Linear layer matmuls on W for xW^T before Mm and Addmm operations.
        assert(len(r.shape) == 2, "Assumption was that matrix would be 2d Linear weights.") # For now assume that it is only 2d matmuls for Linear layers.

        transpose = lambda x: x.T

        if isinstance(r, AddBackwardPromise):
            r.nest_fwd(transpose)
            r.nest_bwd(transpose)
            new_shape = list(r.fwd_shape)
            new_shape[0], new_shape[1] = new_shape[1], new_shape[0]
            r.fwd_shape = tuple(new_shape)
            return r

        return transpose(r)

    @classmethod
    def TransposeBackwardProp(cls, grad_fn, r):
        dim1 = grad_fn._saved_dim0
        dim2 = grad_fn._saved_dim1

        if dim1 == 2**32 - 2:
            dim1 = -2
        if dim2 == 2**32 - 1:
            dim2 = -1

        swapaxes = lambda x: torch.swapaxes(x, dim1, dim2)

        if isinstance(r, AddBackwardPromise):
            r.nest_fwd(swapaxes)
            r.nest_bwd(swapaxes)
            new_shape = list(r.fwd_shape)
            new_shape[dim1], new_shape[dim2] = new_shape[dim2], new_shape[dim1]
            r.fwd_shape = tuple(new_shape)
            return r
        
        return swapaxes(r)

    @classmethod
    def PermuteBackwardProp(cls, grad_fn, r):
        dims = grad_fn._saved_dims
        permute = lambda x: torch.permute(x, dims)

        if isinstance(r, AddBackwardPromise):
            r.nest_fwd(permute)
            r.nest_bwd(permute)
            new_shape = list(r.fwd_shape)
            for old_dim, new_dim in enumerate(dims):
                new_shape[old_dim] = r.fwd_shape[new_dim]
            r.fwd_shape = tuple(new_shape)
            return r
        return permute(r)

    @classmethod
    def ExpandBackwardProp(cls, grad_fn, r):
        upstream_shape = grad_fn._saved_self_sym_sizes
        downstream_shape = r.shape
        assert(len(upstream_shape) == len(downstream_shape), "Expand should not increase number of dimensions.")

        expand_input = [ dim2 if dim1 != dim2 else -1 for dim1, dim2 in zip(upstream_shape, downstream_shape) ]

        def undoExpand(x):
            for i, expand_dim in enumerate(expand_input):
                if expand_dim != -1:
                    x = x.select(i, 0).unsqueeze(i)
            return x

        expand = lambda x: x.expand(*expand_input)

        if isinstance(r, AddBackwardPromise):
            r.nest_fwd(expand)
            r.nest_bwd(undoExpand)
            r.fwd_shape = upstream_shape
            return r

        return undoExpand(r)

    @classmethod
    def MulBackwardProp(cls, grad_fn, r):
        arg1 = grad_fn._saved_self
        arg2 = grad_fn._saved_other

        if isinstance(r, AddBackwardPromise):
            if arg1 is None:
                r.nest_fwd(lambda x: x * arg2)
            else:
                r.setarg(arg1 * arg2)
                if r.complete:
                    r = r.rin
                else:
                    return None # Trigger requeue

        if arg1 is None:
            # Tensor-scalar product, disregard scalar
            return r, 0.0

        denom = arg1.abs() + arg2.abs() + epsilon
        r1 = (arg1.abs() / denom) * r
        r2 = (arg2.abs() / denom) * r

        return renormalize_epsilon(r, r1, r2)

    @classmethod
    def DivBackwardProp(cls, grad_fn, r):
        arg1 = grad_fn._saved_self
        arg2 = grad_fn._saved_other

        if isinstance(r, AddBackwardPromise):
            if arg1 is None:
                r.nest_fwd(lambda x: x / arg2)
            else:
                r.setarg(arg1 / arg2)
                if r.complete:
                    r = r.rin
                else:
                    return None # Trigger requeue

        if arg1 is None:
            # Tensor-scalar product, disregard scalar
            return r, 0.0

        denom = arg1.abs() + (1 / arg2).abs() + epsilon
        r1 = (arg1.abs() / denom) * r
        r2 = ((1 / arg2).abs() / denom) * r

        return renormalize_epsilon(r, r1, r2)

    @classmethod
    def MmBackwardProp(cls, grad_fn, r):
        x = grad_fn._saved_mat1 # i j
        weights = grad_fn._saved_mat2 # j k
        z = torch.matmul(x, weights)
        if isinstance(r, AddBackwardPromise):
            r.setarg(z)
            if r.complete:
                r = r.rin
            else:
                # If this is the first branch of the promise
                return None # Make this trigger a requeue

        i, j = x.shape
        k = weights.shape[1]
        intermediates = torch.einsum("ij, jk -> ijk", x, weights)

        z = z.unsqueeze(1).broadcast_to((i,j,k))

        ratios = intermediates / z

        # return relevance for input and relevance for weight
        return ratios.sum(dim=2, keepdims=True) * r, ratios.sum(dim=0, keepdims=True) * r.T

    @classmethod
    def BmmBackwardProp(cls, grad_fn, r):
        mat1 = grad_fn.saved_self
        mat2 = grad_fn.saved_mat2
        z = torch.matmul(mat1, mat2)
        if isinstance(r, AddBackwardPromise):
            r.setarg(z)
            if r.complete:
                r = r.rin
            else:
                # If this is the first branch of the promise
                return None # Make this trigger a requeue

        b, i, j = mat1.shape
        k = mat2.shape[-1]
        intermediates = torch.einsum("bij, bjk -> bijk", mat1, mat2)

        z = z.unsqueeze(2).broadcast_to((b,i,j,k))

        ratios = intermediates / z

        # return relevance for mat1 and relevance for mat2
        return ratios.sum(dim=2, keepdims=True) * r, ratios.sum(dim=1, keepdims=True) * r.T

    @classmethod
    def NativeLayerNormBackwardProp(cls, grad_fn, r):
        # next_functions will correspond to input, weights, bias
        # We only care about propagating through the input for LayerNorm.
        return r, 0.0, 0.0

    @classmethod
    def IdentityProp(cls, grad_fn, r):
        """Placeholder for any missed operations, or general use for identity-rule operations."""
        return r
    
    @classmethod
    def AccumulateGradProp(cls, grad_fn, r):
        if isinstance(r, AddBackwardPromise):
            r.setarg(grad_fn.variable)
        return 0.0

    @classmethod
    def LRPCheckpointBackwardProp(cls, grad_fn, r):
        if isinstance(r, AddBackwardPromise):
            r.checkpoint(grad_fn)
        else:
            grad_fn.metadata["checkpoint_relevance"] = r
        return r

    @classmethod
    def generate_prop_fcn_map(cls, names: list[str]):
        """Creates a dict to map grad_fn names to their corresponding LRP propagation
        functions"""
        fcn_map = {}
        for name in names:
            fcn_name = name + "Prop"
            if name[0].isdigit():
                fcn_name = name[:-1] + "Prop"
            if (hasattr(cls, fcn_name) and callable(fcn := getattr(cls, fcn_name))
                and not fcn_name.startswith("_")):
                fcn_map[name] = fcn
            else:
                fcn_map[name] = cls.IdentityProp
        return fcn_map