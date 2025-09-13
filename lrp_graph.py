import torch
from torch.autograd.graph import Node
from util import (
    decompose_addmmbackward,
    LRPCheckpoint,
)

def make_graph(hidden_states: torch.Tensor, return_topo_dict: bool = False):
    """Creates an auxiliary graph from the autograd graph starting at hidden_states
    and going backward.

    Returns graph in form of in-adjacency and out-adjacency lists, the set of all fcn names,
    and optionally one of either the mapping between topological index and node or the total
    number of nodes in the graph.

    hidden_states     : model output
    return_topo_dict  : flag for choosing what to return"""

    out_adj = {}
    in_adj = {}

    root = hidden_states.grad_fn
    visited = set()
    topo_stack = []
    update_root = None

    if type(root).__name__ == "AddmmBackward0":
        update_root = [root]

    make_graph_topo_dfs(root, in_adj, out_adj, visited, topo_stack, update_root)
    if return_topo_dict:
        return in_adj, out_adj, set([ type(f).__name__ for f in visited ]), dict(topo_stack), len(topo_stack), update_root
    else:
        return in_adj, out_adj, set([ type(f).__name__ for f in visited ]), None, len(topo_stack), update_root

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

def make_graph_topo_dfs(fcn : Node, in_adj, out_adj, visited, topo_stack, update_root=None):
    if fcn is None or fcn in visited:
        return
    visited.add(fcn)

    if type(fcn).__name__ == "AddmmBackward0":
        # Decompose the function into an Add + Mm, then re-assign its adjacencies.
        decomposed_add = decompose_addmmbackward(fcn)
        if update_root is not None and fcn == update_root[0]:
            update_root[0] = decomposed_add
        if fcn in in_adj:
            # Assign new Add's in-neighbours to the AddMm's in-neighbours.
            in_adj[decomposed_add] = in_adj[fcn]
            for in_neighbour in in_adj[fcn]:
                # Replace all out-edges going to the AddMm to point to the new Add.
                old_fcn_idx = out_adj[in_neighbour].index(fcn)
                out_adj[in_neighbour][old_fcn_idx] = decomposed_add
            
            del in_adj[fcn]
        fcn = decomposed_add

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
    for (child, _) in fcn.next_functions:
        out_adj[fcn].append(child)

        # Crucial that this comes after setting out_adj[fcn]
        if child is None:
            continue
        if child not in in_adj:
            in_adj[child] = []
        in_adj[child].append(fcn)

        make_graph_topo_dfs(child, in_adj, out_adj, visited, topo_stack)
    
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
         [ c.metadata["topo_ind"] if c is not None else None for c in children ]
        for node, children in list(out_adj.items())
    }

    return new_in_adj, new_out_adj

def create_checkpoint_execution_plan(
        checkpoint_inds: list[int],
        in_adj: dict[int, list[int]],
        num_nodes: int,
        stop_node_ind: int,
        prom_inner_node_inds: set[int],
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
