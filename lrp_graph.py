import torch
from util import decompose_addmmbackward

def make_graph(hidden_states: torch.Tensor):
    """Creates an auxiliary graph from the autograd graph starting at hidden_states
    and going backward.
    hidden_states: model output
    Returns graph in form of in-adjacency and out-adjacency lists, and set of all fcn names."""
    out_adj = {}
    in_adj = {}

    root = hidden_states.grad_fn
    visited = set()
    names = set()
    fcns = [ [root] ]
    # idea: dynamically init relevance variables when branching occurs, assign them to the corresponding
    # downstream nodes they should belong to using next_functions and visited table. Requires 2 passes.
    # Perhaps need a last_saved_relevance for each node, in the case when a node is traversed more than once to accumulate relevance.
    # Need all incoming branches to land before we continue, else we need to compute the same downstream paths multiple times for
    # each incoming branch.
    # Modified DFS? Traverse down a path, creating all necessary relevance branches until a node with multiple in-edges is reached.
    # We will need a modified graph as well with in_children for each node to determine the above condition.

    # First pass will:
    # 1. Create in and out adjacency lists.
    # 2. Decompose AddMmBackward0's with AddBackward0 leading back to MmBackward0.
    while fcns:
        new_fcns = []
        for fcn_list in fcns:
            for fcn in fcn_list:

                if fcn is None or fcn in visited:
                    continue

                if fcn.name == "AddMmBackward0":
                    # Decompose the function into an Add + Mm, then re-assign its adjacencies.
                    decomposed_add = decompose_addmmbackward(fcn)
                    # Assign new Add's in-neighbours to the AddMm's in-neighbours.
                    in_adj[decomposed_add] = in_adj[fcn]
                    for in_neighbour in in_adj[fcn]:
                        # Replace all out-edges going to the AddMm to point to the new Add.
                        old_fcn_idx = out_adj[in_neighbour].index(fcn)
                        out_adj[in_neighbour][old_fcn_idx] = decomposed_add
                    del in_adj[fcn]
                    fcn = decomposed_add

                # Assign adjacencies
                if fcn not in out_adj:
                    out_adj[fcn] = []
                for (child, _) in fcn.next_functions:
                    out_adj[fcn].append(child)
                    if child not in in_adj:
                        in_adj[child] = []
                    in_adj[child].append(fcn)

                visited.add(fcn)
                if type(fcn).__name__ not in names:
                    names.add(type(fcn).__name__)

                new_fcns.append([ fcn_tup[0] for fcn_tup in fcn.next_functions ])

        # Iterate
        fcns = new_fcns

    return in_adj, out_adj, names
