import torch
from torch.autograd.graph import Node
from typing import Union
from .util import (
    decompose_addmmbackward,
    decompose_convbackward,
    decompose_addcmulbackward,
)

DECOMPOSABLE_FUNCTIONS = (
    # Nodes that must be decomposed
    "AddmmBackward0",
    "ConvolutionBackward0",
    "AddcmulBackward0",
)

def make_graph(output_tuple_or_tensor: Union[tuple[torch.Tensor], torch.Tensor], params_to_interpret: list[torch.Tensor] = None, return_topo_dict: bool = False):
    """Creates an auxiliary graph from the autograd graph starting at hidden_states
    and going backward.

    Returns graph in form of in-adjacency and out-adjacency lists, the set of all fcn names,
    and optionally one of either the mapping between topological index and node or the total
    number of nodes in the graph.

    params_to_interpret: all model parameter tensors that we wish to obtain relevance maps for, only provided on the first LRP pass, reused for future passes
    hidden_states      : model output
    return_topo_dict   : flag for choosing whether or not to return an {ind: node} dict"""

    out_adj = {}
    in_adj = {}

    if isinstance(output_tuple_or_tensor, torch.Tensor):
        roots = [ output_tuple_or_tensor.grad_fn ]
    elif isinstance(output_tuple_or_tensor, tuple):
        roots = [ output.grad_fn for output in output_tuple_or_tensor ]
    visited = set()
    names = {}
    topo_stack = []
    updated_roots = [ root for root in roots ]
    param_nodes = []

    for root in roots:
        make_graph_topo_dfs(root, in_adj, out_adj, visited, names, topo_stack, updated_roots, params_to_interpret, param_nodes)
    if return_topo_dict:
        return in_adj, out_adj, names, dict(topo_stack), len(topo_stack), updated_roots, param_nodes
    else:
        return in_adj, out_adj, names, None, len(topo_stack), updated_roots, param_nodes

    # idea: dynamically init relevance variables when branching occurs, assign them to the corresponding
    # downstream nodes they should belong to using next_functions and visited table. Requires 2 passes.
    # Need all incoming branches to land before we continue, else we need to compute the same downstream paths multiple times for
    # each incoming branch.
    # Traverse down a path, propagating relevance until a node with multiple in-edges is reached.
    # We will need a modified graph as well with in_children for each node to determine the above condition.

    # First pass will:
    # 1. Create in and out adjacency lists.
    # 2. Decompose AddmmBackward0's with AddBackward0 leading back to MmBackward0.
    # 3. Assign each grad_fn node to its unique integer id based on traversal order

def make_graph_topo_dfs(fcn : Node, in_adj, out_adj, visited, names, topo_stack, updated_roots, params_to_interpret : list[torch.Tensor], param_nodes : list[Node]):
    """
    visited, topo_stack, updated_roots, and param_nodes are all accumulators and must be set and provided by the caller.

    - topo_stack: a list of tuples [ (fcn, topo_ind), ... ] ordered in reverse topological order (desc indices), with the roots at the highest indices.
    - updated_roots: a list of either None or DecomposedNode, indicating if any of the root Nodes were replaced via a decomposition.
    - param_nodes: a list of AccumulateGrad Nodes which match model parameters given by the user to interpret and all EmbeddingBackward0 Nodes.
    """
    if fcn is None or fcn in visited:
        return
    visited.add(fcn)

    if (fcn_name := type(fcn).__name__) in DECOMPOSABLE_FUNCTIONS:
        if fcn_name == "AddmmBackward0":
            # Decompose the function into an Add + Mm.
            decomposed_root = decompose_addmmbackward(fcn)
        elif fcn_name == "ConvolutionBackward0":
            # Decompose the function into an Add + Conv.
            decomposed_root = decompose_convbackward(fcn)
        elif fcn_name == "AddcmulBackward0":
            # Decompose the function into an Add + Mm.
            decomposed_root = decompose_addcmulbackward(fcn)
        else:
            raise ValueError(f"{fcn_name} is marked as needing decomposition but does not have a decomposition handler.")
        
        try:
            fcn_root_ind = updated_roots.index(fcn)
            # Set the updated root to the decomposed root
            updated_roots[fcn_root_ind] = decomposed_root
        except ValueError as e:
            # fcn is not a root
            pass

        if fcn in in_adj:
            # Assign new Add's in-neighbours to the AddMm's in-neighbours.
            in_adj[decomposed_root] = in_adj[fcn]
            for in_neighbour in in_adj[fcn]:
                # Replace all out-edges going to the AddMm to point to the new Add.

                for i, (out_node, out_idx) in enumerate(out_adj[in_neighbour]):
                    if out_node == fcn:
                        out_adj[in_neighbour][i] = (decomposed_root, out_idx)

                # old_fcn_idx = out_adj[in_neighbour].index(fcn)
                # out_adj[in_neighbour][old_fcn_idx] = decomposed_add
            
            del in_adj[fcn]
        fcn = decomposed_root
    elif fcn_name == "AccumulateGrad" and params_to_interpret:
        # Label the input node so during backprop we can use it as an extra stopping condition
        if any(fcn.variable is param for param in params_to_interpret):
            params_to_interpret.remove(fcn.variable)
            param_nodes.append(fcn) # Can't use indices yet since they havent been set, but should be fine
            fcn.metadata["save_relevance"] = True
    elif fcn_name == "EmbeddingBackward0":
        fcn.metadata["save_relevance"] = True
        param_nodes.append(fcn)

    # Add processed node's name to the names set
    if (fcn_name := type(fcn).__name__) not in names:
        names[fcn_name] = 1
    else:
        names[fcn_name] += 1
    # Assign adjacencies
    # NOTE: Out-adjacencies will account for None, but in-adjacencies will not.
    # This is because we use out_adj for verifying number of expected fwd inputs 
    # and relevance outputs, and in_adj for number of relevance inputs landed.
    # None propagates no relevance, so it cannot possibly be an input. Adding it
    # there would cause the logic to hang.
    # However, it is a valid fwd input (or bwd output), so not adding it in out_adj
    # would cause too much mismatch and confusion between number of outputs (prop
    # fcns account for the None's) and expected number of children (which could be
    # less without the None's)
    if fcn not in out_adj:
        out_adj[fcn] = []
    
    for (child, i) in fcn.next_functions:
        out_adj[fcn].append((child, i))

        # Crucial that this comes after setting out_adj[fcn]
        if child is None:
            continue
        if child not in in_adj:
            in_adj[child] = []
        in_adj[child].append(fcn)

        make_graph_topo_dfs(child, in_adj, out_adj, visited, names, topo_stack, updated_roots, params_to_interpret, param_nodes)
    
    # We can directly store the index in each node's metadata dict, so we only need to return
    # the reverse lookup dict from ind to node.
    topo_ind = len(topo_stack)
    fcn.metadata["topo_ind"] = topo_ind
    topo_stack.append((topo_ind, fcn))

def convert_graph_to_index_based(in_adj, out_adj):
    new_in_adj = {
        node.metadata["topo_ind"] :
         [ p.metadata["topo_ind"] for p in parents ]
        for node, parents in list(in_adj.items())
    }
    new_out_adj = {
        node.metadata["topo_ind"] :
         [ (c.metadata["topo_ind"], idx) if c is not None else (None, 0) for (c, idx) in children ]
        for node, children in list(out_adj.items())
    }

    return new_in_adj, new_out_adj

def create_checkpoint_execution_plan(
        checkpoint_inds: list[int],
        in_adj: dict[int, list[int]],
        num_nodes: int,
        stop_node_ind: int,
        prom_inner_node_inds: set[int],
        param_node_inds: list[int] = None,
    ) -> list[int]:
    """Starting at each checkpoint node, finds all nodes ascending from that checkpoint
    to the original hidden_states grad_fn node or any checkpoints in between.

    Returns a list of the found ancestor node indices in reverse topological order.

    Guaranteed to terminate since all nodes originate from hidden_states.
    
    checkpoint_inds  : list of all LRPCheckpoint grad_fn indices in the computation graph
    in_adj           : index-based in-adjacency list of the graph
    num_nodes        : number of unique node indices stored by the topological sort
    stop_node_ind    : unique index of the hidden_states leaf grad_fn node
    prom_inner_node_inds: all node indices corresponding to nodes inside a promise chain
    ind_to_node      : maps index to autograd node, for debugging
    """

    stop_inds = [ stop_node_ind, *checkpoint_inds ]

    if param_node_inds is not None:
        checkpoint_inds += param_node_inds

    # ancestor_nodes[i] will be set to 1 if node i in the topological ordering is an ancestor of any checkpoint
    ancestor_nodes = [ 0 for _ in range(num_nodes) ]
    visited = set()

    for cur_checkpoint in checkpoint_inds:
        stack : list[int] = [ cur_checkpoint ]

        while stack:
            curnode = stack.pop()

            if curnode in visited:
                continue

            if curnode not in prom_inner_node_inds:
                # We don't flag inner nodes of Promises since we should not recompute these propagations,
                # they should be baked into the Promise's fwd() and bwd().
                # We do need to keep traversing through Promise inner nodes, though, so don't just end the branch.
                ancestor_nodes[curnode] = 1

            if curnode != cur_checkpoint and curnode in stop_inds:
                # Condition for stopping traversal on a given branch
                continue

            visited.add(curnode)

            stack = stack + in_adj[curnode]

    # Return in reverse order (default is already according to topo sort)
    # Note that the reverse topological order of a DAG is a topological order of the same DAG with all edge directions flipped
    return [ i for i, flag in list(enumerate(ancestor_nodes))[::-1] if flag == 1 ]
