import time
import torch
import torch.nn.functional as F
from add_backward_promise import AddBackwardPromise
from sum_backward_promise import SumBackwardPromise
from cat_backward_promise import CatBackwardPromise
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
        print(f"{grad_fn}:", end="") 
        if type(grad_fn).__name__ == "AddBackward0":
            if isinstance(r, Promise):
                print(r.rout.nansum(), end=" ")
            else:
                print(r.nansum(), end=" ")
        elif type(grad_fn).__name__ == "AccumulateGrad":
            return res
        if (not isinstance(r, Promise)) and \
                not isinstance(res, Promise) \
                and (not isinstance(res, tuple) or not isinstance(res[0], Promise)):
            rout = r.nansum()
            rins = None
            if isinstance(res, tuple):
                rins = ((res[0].nansum() if isinstance(res[0], torch.Tensor) else res[0]) +
                        (res[1].nansum() if isinstance(res[1], torch.Tensor) else res[1]))
            else:
                rins = res.sum()
            print(rout, rins, end=" ")
        print(memused := torch.cuda.memory_allocated("cuda:0"), memres := torch.cuda.memory_reserved("cuda:0"), memres - memused)
        return res
    return wrapper

def add_node_to_promise_path(func):
    def wrapper(*args, **kwargs):
        r = args[2]
        grad_fn = args[1]
        if isinstance(r, Promise):
            r.add_to_path(grad_fn.metadata["topo_ind"])
            r.fwd_shape = grad_fn._input_metadata[0].shape
            # r.ind_to_node[grad_fn.metadata["topo_ind"]] = grad_fn
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
            promise["rout"] = torch.zeros(r.fwd_shape, device=r.rout.device, dtype=r.rout.dtype) # Placeholder for shape

        traversal_ind = grad_fn.metadata["topo_ind"]

        promise1 = AddBackwardPromise(promise, traversal_ind, 0)
        promise2 = AddBackwardPromise(promise, traversal_ind, 1)

        promise1.other_branch = promise2
        promise2.other_branch = promise1

        if isinstance(r, Promise):
            r.arg_node_ind = traversal_ind
            r.children.append((promise1, promise2))

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
            promise["rout"] = torch.zeros(r.fwd_shape, device=r.rout.device, dtype=r.rout.dtype) # Placeholder for shape
        
        traversal_ind = grad_fn.metadata["topo_ind"]

        promise = SumBackwardPromise(promise, traversal_ind, getattr(grad_fn, "_saved_dim"), getattr(grad_fn, "_saved_keepdim"))

        if isinstance(r, Promise):
            r.arg_node_ind = traversal_ind
            r.children.append(promise)

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
    def CatBackwardProp(cls, grad_fn, r):
        if isinstance(r, Promise):
            rout_placeholder = torch.zeros(r.fwd_shape, device=r.rout.device, dtype=r.rout.dtype)
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

            prev_promise_branch = None
            promise_branches = []
            # Since you can cat an arbitrary number of tensors, we need to get a bit creative and turn other_branch into a cyclic connection
            for i in range(num_args):
                new_branch = CatBackwardPromise(promise, traversal_ind, grad_fn._saved_dim, i)
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

    @classmethod
    @output_relevances
    @add_node_to_promise_path
    def MaxBackwardProp(cls, grad_fn, r):
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

    @classmethod
    @output_relevances
    @add_node_to_promise_path
    def ViewBackwardProp(cls, grad_fn, r):
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
    @output_relevances
    @add_node_to_promise_path
    def UnsafeViewBackwardProp(cls, grad_fn, r):
        return cls.ViewBackwardProp(grad_fn, r)

    @classmethod
    @output_relevances
    @add_node_to_promise_path
    def ReshapeBackwardProp(cls, grad_fn, r):
        return cls.ViewBackwardProp(grad_fn, r)

    @classmethod
    @output_relevances
    @add_node_to_promise_path
    def ReshapeAliasBackwardProp(cls, grad_fn, r):
        return cls.ViewBackwardProp(grad_fn, r)

    @classmethod
    @output_relevances
    @add_node_to_promise_path
    def SliceBackwardProp(cls, grad_fn, r):
        def create_pad(node):
            # Assumes the index corresponding to _saved_start in the forward pass is non-negative.
            # If it was negative-indexed, i.e. x[-i:] autograd saves the index as INT_MAX - i.
            upstream_shape = node._saved_self_sym_sizes
            sliced_dim = node._saved_dim
            start = node._saved_start # TODO: Come back to take care of the negative index case.
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

            return tuple(pad)

        if isinstance(r, Promise):
            def fwd_factory(node):
                expected_fwd_shape = node._input_metadata[0].shape
                def fwd_slice(x):
                    return torch.ops.aten.slice(x, dim := node._saved_dim, start := node._saved_start, start + expected_fwd_shape[dim])
                return fwd_slice
            
            def bwd_factory(node):
                def bwd_slice(x):
                    return F.pad(x, create_pad(node))
                return bwd_slice

            r.nest_fwd(fwd_factory, grad_fn.metadata["topo_ind"])
            r.nest_bwd("self", grad_fn.metadata["topo_ind"])
            return r
        return F.pad(r, create_pad(grad_fn))

    @classmethod
    @output_relevances
    @add_node_to_promise_path
    def IndexBackwardProp(cls, grad_fn, r):
        if isinstance(r, Promise):
            def fwd_factory(node):
                def fwd_index(x):
                    return torch.ops.aten.index(x, [ torch.tensor(x) if x is not None else None for x in node._saved_indices ])
                return fwd_index

            r.nest_fwd(fwd_factory, grad_fn._metadata["topo_ind"])
            r.nest_bwd("self", grad_fn.metadata["topo_ind"])
            return r
        return grad_fn(r)

    @classmethod
    @output_relevances
    @add_node_to_promise_path
    def SelectBackwardProp(cls, grad_fn, r):

        def undoSelect(node, x):
            upstream_shape = node._saved_self_sym_sizes
            dim = grad_fn._saved_dim
            idx = node._saved_index
            out = torch.zeros(upstream_shape, dtype=x.dtype, device=x.device)
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

        if isinstance(r, Promise):
            r.nest_fwd("self", grad_fn.metadata["topo_ind"])
            r.nest_bwd("self", grad_fn.metadata["topo_ind"])
            return r

        return grad_fn(r)

    @classmethod
    @output_relevances
    @add_node_to_promise_path
    def TransposeBackwardProp(cls, grad_fn, r):
        if isinstance(r, Promise):

            r.nest_fwd("self", grad_fn.metadata["topo_ind"])
            r.nest_bwd("self", grad_fn.metadata["topo_ind"])
            return r
        
        return grad_fn(r)

    @classmethod
    @output_relevances
    @add_node_to_promise_path
    def PermuteBackwardProp(cls, grad_fn, r):
        if isinstance(r, Promise):

            def fwd_factory(node):
                def fwd_permute(x: torch.Tensor):
                    return x.permute(node._saved_dims)
                return fwd_permute

            r.nest_fwd(fwd_factory, grad_fn.metadata["topo_ind"])
            r.nest_bwd("self", grad_fn.metadata["topo_ind"])
            return r
        return grad_fn(r)

    @classmethod
    @output_relevances
    @add_node_to_promise_path
    def ExpandBackwardProp(cls, grad_fn, r):

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
    def UnsqueezeBackwardProp(cls, grad_fn, r):
        if isinstance(r, Promise):
            r.nest_fwd("self", grad_fn.metadata["topo_ind"])
            r.nest_bwd("self", grad_fn.metadata["topo_ind"])
            return r

        return grad_fn(r)
    
    @classmethod
    @output_relevances
    @add_node_to_promise_path
    def NegBackwardProp(cls, grad_fn, r):
        return r
    
    @classmethod
    @output_relevances
    @add_node_to_promise_path
    def CloneBackwardProp(cls, grad_fn, r):
        return r

    @classmethod
    @output_relevances
    @add_node_to_promise_path
    def MulBackwardProp(cls, grad_fn, r):
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
                r.setarg(arg1.detach() * arg2.detach(), grad_fn, lambda fcn: fcn._saved_self.detach() * fcn._saved_other.detach())
                if r.complete:
                    r = r.rin
                else:
                    return r # We will check if the promise is complete in the graph traversal

        if arg1 is None:
            # Tensor-scalar product, disregard scalar
            return r, 0.0
        
        arg1, arg2 = arg1.detach(), arg2.detach()

        denom = arg1.abs() + arg2.abs() + epsilon

        # Reduce relevance to denom’s shape instead of expanding denom to r
        factor = r.sum_to_size(*denom.shape)  # requires PyTorch ≥ 2.0
        r1 = (arg1.abs() / denom) * factor
        r2 = (arg2.abs() / denom) * factor

        r1, r2 = renormalize_epsilon(r, r1, r2)
        return r1, r2

    @classmethod
    @output_relevances
    @add_node_to_promise_path
    def DivBackwardProp(cls, grad_fn, r):
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
                r.setarg(arg1.detach() / arg2.detach(), grad_fn, lambda fcn: fcn._saved_self.detach() / fcn._saved_other.detach())
                if r.complete:
                    r = r.rin
                else:
                    return r # We will check if the promise is complete in the graph traversal

        if arg1 is None:
            # Tensor-scalar product, disregard scalar
            return r, 0.0
        
        arg1, arg2 = arg1.detach(), arg2.detach()

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
        x = grad_fn._saved_self.detach() # s d
        weights = grad_fn._saved_mat2.detach() # d o

        #### TODO: (Aug 21, 2025) There is an overlooked case of when one of x, weights is None because one of them does not have requires_grad=True
        # However, this is most likely mitigated if the model is in training mode / using requires_grad as a whole.
        # Priority right now is getting the deterministic execution plan run mode working, can circle back to this later
        # because it will likely require some reworking on Promises as well. See the note I left in promise.py.

        #### TODO: Add an ablation for Attn-LRP vs traditional LRP

        z = x @ weights # s o
        z_stabilized = z + epsilon * z.sign()

        if isinstance(r, Promise):
            r.setarg(z, grad_fn, lambda fcn: fcn._saved_self.detach() @ fcn._saved_mat2.detach())
            if r.complete:
                r = r.rin
            else:
                # If this is the first branch of the promise
                return r # We will check if the promise is complete in the graph traversal
        
        # Optimized to avoid creating large intermediates
        tmp = r / z_stabilized   # s o

        rin_input = x * (tmp @ weights.t()) # s d
        rin_input = rin_input / 2

        rin_weight = weights * (x.t() @ tmp) # s o
        rin_weight = rin_weight / 2

        # if (next_fcn := grad_fn.next_functions[1][0]) is not None \
        #         and (type(next_fcn).__name__ == "AccumulateGrad"
        #              or type(next_fcn.next_functions[0][0]).__name__ == "AccumulateGrad"):
        #     return torch.einsum('sdo,so->sd', ratio, r), torch.zeros_like(weights)

        # propagate relevance in parallel for input and weight, as specified by A.3.2 (Bilinear matmul) of AttnLRP: https://arxiv.org/pdf/2402.05602
        return rin_input, rin_weight

    @classmethod
    @output_relevances
    @add_node_to_promise_path
    def BmmBackwardProp(cls, grad_fn, r):
        mat1 = grad_fn._saved_self.detach() # b s d
        mat2 = grad_fn._saved_mat2.detach() # b d o
        z = torch.bmm(mat1, mat2) # b s o

        # assert r.shape == z.shape, f"r shape {r.shape} must match z shape {z.shape}"

        z_stabilized = z + epsilon * z.sign()

        if isinstance(r, Promise):
            r.setarg(z, grad_fn, lambda fcn: fcn._saved_self.detach() @ fcn._saved_mat2.detach())
            if r.complete:
                r = r.rin
            else:
                # If this is the first branch of the promise
                return r # We will check if the promise is complete in the graph traversal
            
        # Same optimization as MmBackward
        tmp = r / z_stabilized # b s o

        rin_mat1 = mat1 * (tmp @ mat2.permute(0, 2, 1))
        rin_mat1 = rin_mat1 / 2

        rin_mat2 = mat2 * (mat1.permute(0, 2, 1) @ tmp)
        rin_mat2 = rin_mat2 / 2

        # if (next_fcn := grad_fn.next_functions[1][0]) is not None \
        #         and (type(next_fcn).__name__ == "AccumulateGrad"
        #              or type(next_fcn.next_functions[0][0]).__name__ == "AccumulateGrad"):
        #     return torch.einsum('bsdo,bso->bsd', ratio, r), torch.zeros_like(mat2)

        # propagate relevance in parallel for mat1 and mat2
        return rin_mat1, rin_mat2

    @classmethod
    @output_relevances
    @add_node_to_promise_path
    def NativeLayerNormBackwardProp(cls, grad_fn, r):

        def layerNorm(fcn):
            x = fcn._saved_input.detach()
            mean = fcn._saved_result1.detach()
            gamma = fcn._saved_weight.detach()
            beta = fcn._saved_bias.detach()
            rec_stddev = fcn._saved_result2.detach()
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
        return r, 0.0, 0.0
    
    @classmethod
    @output_relevances
    @add_node_to_promise_path
    def GeluBackwardProp(cls, grad_fn, r):
        if isinstance(r, Promise):
            r.setarg(F.gelu(grad_fn._saved_self.detach()), grad_fn, lambda fcn: torch.nn.GELU(fcn._saved_self.detach()))
            if r.complete:
                r = r.rin
        return r
    
    @classmethod
    @output_relevances
    @add_node_to_promise_path
    def SiluBackwardProp(cls, grad_fn, r):
        if isinstance(r, Promise):
            def apply_silu(node):
                return F.silu(node._saved_self)
            r.setarg(apply_silu(grad_fn), grad_fn, apply_silu)
            if r.complete:
                r = r.rin
        return r
    
    @classmethod
    @output_relevances
    @add_node_to_promise_path
    def SoftmaxBackwardProp(cls, grad_fn, r):
        result = grad_fn._saved_result.detach()
        if isinstance(r, Promise):
            r.setarg(result, grad_fn, "_saved_result")
            if r.complete:
                r = r.rin
        
        rin = result * (r - result * r.sum(dim=-1, keepdim=True))
        return renormalize_epsilon(r, rin, torch.zeros_like(rin))[0]
    
    @classmethod
    @output_relevances
    @add_node_to_promise_path
    def PowBackwardProp(cls, grad_fn, r):
        if isinstance(r, Promise):
            def get_result(node):
                return node._saved_self.pow(node._saved_exponent)
            r.setarg(get_result(grad_fn), grad_fn, get_result)
            if r.complete:
                r = r.rin
        return r
    
    @classmethod
    @output_relevances
    @add_node_to_promise_path
    def RsqrtBackwardProp(cls, grad_fn, r):
        if isinstance(r, Promise):
            def get_result(node):
                return node._saved_self.pow(node._saved_exponent).reciprocal()
            r.setarg(get_result(grad_fn), grad_fn, get_result)
            if r.complete:
                r = r.rin
        return r
    
    @classmethod
    @output_relevances
    @add_node_to_promise_path
    def IdentityProp(cls, grad_fn, r):
        """Placeholder for any missed operations, or general use for identity-rule operations."""
        num_rel_outs = len(grad_fn.next_functions)

        if isinstance(r, Promise):
            if hasattr(grad_fn, "_saved_result"):
                r.setarg(grad_fn._saved_result.detach(), grad_fn, "_saved_result")
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
            r.setarg(grad_fn.variable.detach(), grad_fn, "variable")
        return 0.0

    @classmethod
    @output_relevances
    @add_node_to_promise_path
    def LRPCheckpointBackwardProp(cls, grad_fn, r):
        saved_input = grad_fn.saved_tensors[0].detach()
        if isinstance(r, Promise):
            r.setarg(saved_input, grad_fn, lambda fcn: fcn.saved_tensors[0])
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

        def bwd_factory(node):
            idxs : torch.Tensor = node._saved_indices
            weights = node.next_functions[0][0].variable # assumption here

            def bwd_embedding(x):
                out = torch.zeros_like(weights)
                if idxs.dim() == 1:
                    out.index_add_(0, idxs, x)
                else:
                    for inds in idxs:
                        out.index_add_(0, inds, x)
                return out
            
            return bwd_embedding
        
        def fwd_factory(node):
            idxs : torch.Tensor = node._saved_indices
            expected_fwd_shape = node._input_metadata[0].shape
            def fwd_embedding(x):
                x[idxs.flatten()].reshape(expected_fwd_shape)
            
            return fwd_embedding

        if isinstance(r, Promise):
            r.nest_fwd(fwd_factory, grad_fn.metadata["topo_ind"])
            r.nest_bwd(bwd_factory, grad_fn.metadata["topo_ind"])
            return r

        return bwd_factory(grad_fn)(r)

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
