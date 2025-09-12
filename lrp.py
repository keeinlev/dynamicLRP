import torch
import time
from transformers.modeling_outputs import SequenceClassifierOutput
from torch.autograd.graph import Node
from lrp_graph import (
    make_graph,
    convert_graph_to_index_based,
    create_checkpoint_execution_plan,
)
from lrp_prop_fcns import LRPPropFunctions
from add_backward_promise import AddBackwardPromise
from dummy_promise import compound_promises
from promise import Promise
from util import (
    create_checkpoint,
    DEBUG,
)
from typing import Callable, Union

def checkpoint_hook(module, input, output):
    return create_checkpoint(output)

def checkpoint_pre_hook(module, input):
    return tuple( create_checkpoint(x) for x in input )

def seq_class_lrp_hook(module, input, output, print_cond = False):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logits : torch.Tensor = None
    
    if isinstance(output, SequenceClassifierOutput):
        logits = output.logits
    elif isinstance(output, tuple[torch.Tensor]):
        logits = output[1]
    else:
        raise ValueError("Expected a model output of type tuple[Tensor] or SequenceClassifierOutput for running LRP. Is the model being used a Sequence Classifier model?")
    
    lrp_checkpoints, _, _, visited = lrp_engine(logits.to(device))
    with open("lrp_outputs.pt", "wb") as fileOut:
        torch.save(lrp_checkpoints, fileOut)
    
    if print_cond:
        print(list(visited))

    return output


def lrp_engine(
        hidden_states : torch.Tensor,
        out_adj_list: Union[set[Node], set[int]] = None,
        topo_exec_order : list[int] = None,
        fcn_map : dict[str, Callable] = None,
        no_recompile = False,
    ):

    start_time = time.time()

    with torch.no_grad():
        # Create the first relevance layer via max logit.
        m = hidden_states.max(-1)
        relevance = torch.zeros_like(hidden_states)
        if m.indices.dim() == 0:
            relevance[0] = 1.0
        else:
            s = hidden_states.shape[-2]
            dim = relevance.dim()
            
            for i, inds in enumerate(m.indices):
                if dim == 2:
                    relevance[list(range(s)),inds] = torch.ones_like(m.values[0])
                elif dim == 3:
                    relevance[i,list(range(s)),inds] = torch.ones_like(m.values[0])
                else:
                    raise ValueError(f"LRP Engine received model outputs of unexpected size {dim}.")

        if out_adj_list is None or topo_exec_order is None or fcn_map is None:
            in_adj_list, out_adj_list, names, ind_to_node, num_nodes = make_graph(hidden_states, True)

            Promise.ind_to_node = ind_to_node

            input_tracker : dict[Node, list] = { k : [] for k in list(in_adj_list.keys()) }
            checkpoints = list(filter(lambda k: type(k).__name__ == "LRPCheckpointBackward", list(in_adj_list.keys())))
            num_checkpoints_reached = 0

            fcn_map = LRPPropFunctions.generate_prop_fcn_map(names)

            visited = set()

            # Setup the first iteration
            input_tracker[hidden_states.grad_fn] = [ relevance ]
            stack : list[Node] = [hidden_states.grad_fn]
            in_adj_list[hidden_states.grad_fn] = []
            nodes_pending = { k : len(v) for k, v in list(in_adj_list.items()) }

            promise_queue : list[Node] = []

            promise_traversal_stack : list[Node] = []
            promise_traversal_mode = False

            promise_fulfillment_mode = False

            while (stack or promise_traversal_stack or promise_queue) and num_checkpoints_reached < len(checkpoints):
                
                curnode = None

                ####### RUN MODE DETERMINATION
                
                # Decide where we should take curnode from
                if promise_queue and any(fcn.metadata["promise"]["complete"] for fcn in promise_queue):
                    # Search for the first complete promise in the queue.
                    curnode = next(( fcn for fcn in promise_queue if fcn.metadata["promise"]["complete"] ))
                    idx = promise_queue.index(curnode)
                    promise_queue = promise_queue[:idx] + promise_queue[idx + 1:]
                    promise_traversal_mode = False
                    promise_fulfillment_mode = True

                elif promise_queue and any(nodes_pending[fcn] == 0 and "pre_promise" in fcn.metadata and
                                        all(parent.complete for parent in fcn.metadata["pre_promise"].parents)
                                        for fcn in promise_queue):
                    # Promises that come from the pre_promise flow, they should be ready but they were created
                    # in promise traversal mode, so their in-relevances are not yet calculated.
                    curnode = next(( fcn for fcn in promise_queue if nodes_pending[fcn] == 0 and 
                                    all(parent.complete for parent in fcn.metadata["pre_promise"].parents) ))
                    idx = promise_queue.index(curnode)
                    promise_queue = promise_queue[:idx] + promise_queue[idx + 1:]

                    # We can process these like normal actually
                    promise_traversal_mode = False
                    promise_fulfillment_mode = False

                    # It's possible that this Node was already waiting on a Promise too, in the diamond deadlock case
                    # If so, just remove the other link to the same promise for cleanliness
                    if "promise" in curnode.metadata:
                        assert curnode.metadata["promise"] == curnode.metadata["pre_promise"].promise, \
                            "Node has both promise and pre_promise metadata fields, but the promises are not equal, traversal logic error."
                        del curnode.metadata["promise"]

                elif promise_traversal_stack:
                    # Second priority is promise traversal, which overrides the requirement for all inputs to land
                    # before traversing a node. However, the promise will not have its rins computed until 
                    curnode = promise_traversal_stack[0]
                    promise_traversal_stack = promise_traversal_stack[1:]
                    promise_traversal_mode = True
                    promise_fulfillment_mode = False

                elif stack:
                    # Fallback to main stack
                    curnode = stack[0]
                    stack = stack[1:]
                    promise_traversal_mode = False
                    promise_fulfillment_mode = False

                ####### END RUN MODE DETERMINATION


                ####### INPUT MERGING

                curnode_inputs = input_tracker[curnode]

                visited.add(curnode) # For debugging

                # According to next_functions
                children = out_adj_list[curnode]

                if not promise_fulfillment_mode:

                    # Categorize all inputs into either pending promises, complete promises, or tensors
                    pending_promise_inputs = []
                    complete_promise_inputs = []
                    tensor_inputs = []
                    for input_ in curnode_inputs:
                        if isinstance(input_, torch.Tensor):
                            tensor_inputs.append(input_)
                        elif isinstance(input_, Promise) and input_.complete:
                            complete_promise_inputs.append(input_)
                        elif isinstance(input_, Promise):
                            pending_promise_inputs.append(input_)
                        elif input_ == 0.0:
                            continue
                        else:
                            print(input_)
                            raise ValueError(f"Expected relevance input to Node {curnode} to be type Promise or Tensor, but got {type(input_)} instead.")
            
                    if not complete_promise_inputs and not pending_promise_inputs and not tensor_inputs:
                        continue
            
                    # Aggregate all inputs into one Tensor or Promise
                    curnode_in_rel = sum(tensor_inputs) + sum([ p.rin for p in complete_promise_inputs ])
                    if pending_promise_inputs:
                        # In promise traversal mode this will be True
                        agg_promises = compound_promises(pending_promise_inputs, curnode.metadata["topo_ind"], promise_traversal_mode, promise_traversal_mode)
                        if curnode_in_rel != 0:
                            curnode_in_rel = agg_promises + curnode_in_rel
                        else:
                            curnode_in_rel = agg_promises
                else:
                    # In promise fulfillment mode, use the completed promise's rin for traversing curnode.
                    curnode_in_rel = curnode.metadata["promise"]["rins"][curnode.metadata["promise_idx"]]
                    if not DEBUG:
                        Promise.clear_args_and_rout_raw(curnode.metadata["promise"])
                    del curnode.metadata["promise"], curnode.metadata["promise_idx"]

                ####### END INPUT MERGING


                ####### PRE-PROMISE RETRIEVAL

                if not promise_traversal_mode and "pre_promise" in curnode.metadata:
                    # We have already traversed a promise tree, but have not calculated its bwd,
                    # since it was done in promise traversal mode.
                    pre_promise : Promise = curnode.metadata["pre_promise"]

                    assert pre_promise.ready, f"Pre-promise at {curnode} was assumed to be ready but was not."

                    if not pre_promise.complete:
                        if isinstance(curnode_in_rel, Promise):
                            # If there is still pending promises at this node, try to complete them via the aggregate promise.
                            # In the case this completes and propagates relevance down, we will have to pick up from the tail nodes
                            # of the aggregate promise.
                            curnode_in_rel.children.append(pre_promise)

                            # This may also actually be a different promise than the pre-promise's original parent, due to
                            # the promise aggregation. Therefore, we must check if the parent-child relationship now flows through
                            # the aggregate promise and not the promise that caused Promise Traversal Mode.
                            # While there is no issue without doing this in first pass, it is needed for the deterministic pass.
                            if curnode_in_rel not in pre_promise.parents:
                                # Due to the pre-promise construction, we know there is only ever one parent, so can safely overwrite.
                                pre_promise.parents = [ curnode_in_rel ]
                            curnode_in_rel.setarg(pre_promise.op_result)
                        else:
                            # Manually set the rout and trigger the promise to finish the backward prop.
                            pre_promise.accumulate_rout(curnode_in_rel)
                            pre_promise.trigger_promise_completion()

                    if not DEBUG:
                        curnode.metadata["pre_promise"].clear_args_and_rout()
                    del curnode.metadata["pre_promise"]

                    tail_nodes = list(pre_promise.promise["tail_nodes"])
                    if curnode in tail_nodes:
                        tail_nodes.remove(curnode)
                    if tail_nodes:
                        # Don't know if the promise is complete yet, so put on the promise queue.
                        # If any are done, they will be traversed with priority.
                        promise_queue += tail_nodes
                        continue
                    else:
                        # If the pre-promise is a singleton, i.e. the node is the tail of its own pre-promise,
                        # just collect the computed rin and re-traverse this node with a tensor rin input like normal.
                        curnode_in_rel = pre_promise.rin

                ####### END PRE-PROMISE RETRIEVAL


                if promise_traversal_mode:
                    # We want to save this so later we'll know we've already traversed this node.
                    curnode.metadata["pre_promise"] = curnode_in_rel
                elif not DEBUG:
                    # We can free up some memory because we will no longer need to access these inputs
                    # Chained promises will maintain their relationships via their class instance members.
                    if nodes_pending[curnode] == 0:
                        del input_tracker[curnode]


                ####### PROPAGATION FCN AND PROMISE QUEUE HANDLING

                # Call the propagation function for the node
                try:
                    curnode_outputs = fcn_map[type(curnode).__name__](curnode, curnode_in_rel)
                except Exception as e:
                    print(e)
                    # raise e
                    return curnode, curnode_in_rel, in_adj_list, out_adj_list

                if isinstance(curnode_outputs, Promise) and curnode_outputs.arg is not None and not curnode_outputs.complete:
                    # Node is waiting on Promise to be completed, add to promise queue and come back later.
                    curnode.metadata["promise"] = curnode_outputs.promise
                    curnode.metadata["promise_idx"] = 0 if not isinstance(curnode_outputs, AddBackwardPromise) else curnode_outputs.idx
                    promise_queue.append(curnode)
                    continue

                ####### END PROPAGATION FCN AND PROMISE QUEUE HANDLING


                ####### OUTPUT PROPAGATION TO CHILDREN

                #   end
                # 1  2  3
                #  \ | /
                #    |
                # rel_out
                #    |
                #    0
                #  start

                # curnode_outputs = prop_fcn(node_0, rel_out) -> rel_in_1, rel_in_2, rel_in_3


                # Children may contain None, like grad_fn.next_functions, to keep integrity of input tracking
                if len(children) == 0 or all(child is None for child in children):
                    continue
                elif len(children) == 1:
                    curnode_outputs = [ curnode_outputs ]
                    
                elif len(children) != len(curnode_outputs):
                    raise ValueError(f"Mismatch: {len(children)} children but {len(curnode_outputs)} outputs from {curnode}.")


                # Update child inputs
                for i, child in enumerate(children):
                    if child is None:
                        # Discard the input (it shouldn't have value anyway), if it's a promise make it a zero-promise
                        if isinstance(curnode_outputs[i], Promise):
                            curnode_outputs[i].setarg(0.0, curnode, lambda node: 0.0)
                        continue
                    input_tracker[child].append(curnode_outputs[i])
                    nodes_pending[child] -= 1
                    assert nodes_pending[child] >= 0, f"Negative pending count for node {child}"
                    assert len(input_tracker[child]) <= len(in_adj_list[child]), \
                        f"Too many inputs landed for {child}"

                ####### END OUTPUT PROPAGATION TO CHILDREN


                ####### SORTING CHILDREN TO WHICH STACK THEY SHOULD GO TO

                # Collect children who now have all their inputs or that have promise(s) depending on them.
                ready_children : list[Node] = [] # All children who have all their inputs landed
                promise_depends_on : list[Node] = [] # All children who do not have all their inputs landed but have at least one incomplete promise input landed.
                for i, child in enumerate(children):
                    if child is None:
                        continue
                    if nodes_pending[child] == 0 and child not in promise_queue:
                        ready_children.append(child)
                    elif isinstance(curnode_outputs[i], Promise) and not curnode_outputs[i].complete and "pre_promise" not in child.metadata:
                        promise_depends_on.append(child)

                promise_traversal_stack = promise_depends_on + promise_traversal_stack
                stack = ready_children + stack
                num_checkpoints_reached = sum([ "checkpoint_relevance" in checkpoint.metadata for checkpoint in checkpoints])

                ####### END SORTING CHILDREN TO WHICH STACK THEY SHOULD GO TO

            end_time = time.time()
            print(f"propagation took {end_time - start_time} seconds")

            # Checking conservation holds across the entire propagation
            # The frontier includes:
            # a) true leaf nodes (no children)
            # b) nodes which received inputs but were never traversed due to computation ending early

            frontier = [ node 
                        for node, out_nodes in list(out_adj_list.items())
                        if len(out_nodes) == 0
                    ]

            frontier += [ node for node in stack if input_tracker[node] ]

            frontier = list(set(frontier))

            # Tally the total relevance at the frontier
            total_frontier_in = 0.0
            for node in frontier:
                total_in = 0.0
                if node not in input_tracker:
                    continue
                for input_ in input_tracker[node]:
                    if isinstance(input_, AddBackwardPromise):
                        if input_.complete:
                            total_in += input_.rin.sum()
                        else:
                            continue
                    elif isinstance(input_, torch.Tensor):
                        total_in += input_.sum()
                total_frontier_in += total_in
            print(total_frontier_in)

            # Checkpoints sorted in desc because they are indexed in the order that we save them in (going backwards).
            checkpoints = sorted(checkpoints, key=lambda c: c.metadata["checkpoint_ind"], reverse=True)
            checkpoint_vals = [ checkpoint.metadata["checkpoint_relevance" ] for checkpoint in checkpoints ]


            # If we want to reuse the graph, we need to re-define it in terms of the topological sort order.
            # Convert autograd-based Node graph into topo_ind graph
            in_adj_list, out_adj_list = convert_graph_to_index_based(in_adj_list, out_adj_list)

            # Filter down the indices of nodes to only what is necessary for computing checkpoints
            topo_exec_order = create_checkpoint_execution_plan(
                [ checkpoint.metadata["topo_ind"] for checkpoint in checkpoints ],
                in_adj_list,
                num_nodes,
                hidden_states.grad_fn.metadata["topo_ind"],
                Promise.all_inner_nodes,
                ind_to_node=ind_to_node
            )

            if not DEBUG:
                Promise.clear_all()
            
            Promise.repair_all_parent_child_connections()

            return checkpoint_vals, out_adj_list, topo_exec_order, fcn_map
        else:
            # Run execution plan made from first pass
            # All adjacency lists and node orders are in terms of topological indices now

            # Need to re-map the graph to all the indices based on topo sort
            _, _, _, ind_to_node, _ = make_graph(hidden_states, True)

            Promise.ind_to_node = ind_to_node

            checkpoints = list(filter(lambda node: type(node).__name__ == "LRPCheckpointBackward", list(ind_to_node.values())))

            input_frontier : dict[int, list[torch.Tensor]] = { ind : 0.0 for ind in topo_exec_order }

            # Setup first iteration
            input_frontier[hidden_states.grad_fn.metadata["topo_ind"]] = relevance


            if no_recompile:
                # First pre-process all Promises by setting every leaf Promise arg
                leaf_promise: Promise
                for leaf_promise in Promise.leaf_promises:
                    if leaf_promise.arg_node_ind is not None:
                        arg_node_grad_fn = ind_to_node.get(leaf_promise.arg_node_ind)

                        leaf_promise.retrieve_and_set_new_arg(arg_node_grad_fn)
                    else:
                        # Case where leaf Promise ended in a None Node
                        leaf_promise.setarg(0.0)
            
                # All promises that are required for the checkpoints should be ready now, as
                # we have set the args of all leaf promises. Now all we need to do is call the
                # bwd() functions as we traverse by each promise.

            fulfilled_promises : set[Promise] = set()

            for node_ind in topo_exec_order:
                iter_start_time = time.time()
                node = ind_to_node[node_ind]

                rel = input_frontier[node_ind]

                if (promise := Promise.start_nodes_to_promise.get(node_ind)) is not None and isinstance(promise, Promise) \
                    and not promise.parents and not no_recompile:

                    # When we hit a Node that starts a Promise, and the Promise is the root of its Promise tree,
                    # we need to update its entire Promise tree's rout values to be of the correct shapes, since
                    # they may have changed with the input.
                    # First set this Promise's rout to rel, as this is correct for sure.
                    promise.set_rout(rel)
                    # Now call the recursive shape propagation function
                    leaf_promises = set()
                    promise.propagate_fwd_shape(rel.shape, leaf_promises)

                    # And of course do this for the other branch too, if it exists
                    if promise.other_branch is not None:
                        promise.other_branch.propagate_fwd_shape(rel.shape, leaf_promises)

                    # Now do the arg fwd-prop from the leaf Promises of just THIS Promise tree.
                    leaf_promise: Promise
                    for leaf_promise in list(leaf_promises):
                        if leaf_promise.arg_node_ind is not None:
                            arg_node_grad_fn = ind_to_node[leaf_promise.arg_node_ind]

                            try:
                                leaf_promise.retrieve_and_set_new_arg(arg_node_grad_fn)
                            except RuntimeError as e:
                                print(node_ind, leaf_promise.fwd_shape)
                                raise e
                        else:
                            # Case where leaf Promise ended in a None Node
                            leaf_promise.setarg(0.0, fwd_only=True, recompile=False)

                if isinstance(promise, Promise) and promise not in fulfilled_promises and \
                        promise.start_ind != promise.arg_node_ind:
                    if not promise.ready:
                        print(node_ind, promise.start_ind, promise.arg_node_ind)
                    assert promise.ready, f"During running of execution plan, promise was missing arg(s) {promise.promise}"

                    # Prep the promise
                    promise.set_rout(rel)
                    if isinstance(promise.rout, float):
                        print(promise.promise, promise.start_ind)
                        raise TypeError
                    promise.compute_rins()

                    # Propagate the rins
                    if promise.arg_node_ind in input_frontier:
                        # Sometimes one of the arg nodes isn't actually a checkpoint ancestor, so we only care about the other one.
                        res = promise.bwd(promise.rin)

                        # Assign rin to the end node of the promise
                        input_frontier[promise.arg_node_ind] += res
                        fulfilled_promises.add(promise)

                    # Do the same for other branch if applicable
                    if (other := promise.other_branch) is not None and other not in fulfilled_promises \
                            and other.arg_node_ind in input_frontier:
                        res2 = other.bwd(other.rin)

                        input_frontier[other.arg_node_ind] += res2
                        fulfilled_promises.add(other)
                    
                    promise.set_complete()
                else:
                    try:
                        res = fcn_map[type(node).__name__](node, rel)
                    except RuntimeError as e:
                        print(node_ind, [ idx for idx in topo_exec_order if node_ind in out_adj_list[idx] ], input_frontier[node_ind].shape)
                        raise e
                    # Distribute the relevance the same way as first pass
                    if not isinstance(res, tuple):
                        res = [ res ]

                    for i, child in enumerate(out_adj_list[node_ind]):
                        if child is None or child not in input_frontier or (isinstance(res[i], float) and res[i] == 0.0):
                            continue

                        input_frontier[child] += res[i]

                # Move the frontier forward
                # Since we are iterating over a topological ordering, when we are a certain node, we are certain
                # that all of its dependencies have been visited at that point, so we can safely delete its inputs.
                del input_frontier[node_ind]

                iter_time = time.time() - iter_start_time
                if iter_time > 1.5 / len(topo_exec_order) and DEBUG:
                    print(f"Node {node_ind}, {node} took {iter_time}s")


            end_time = time.time()
            print(f"propagation took {end_time - start_time} seconds")

            # Checkpoints sorted in desc because they are indexed in the order that we save them in (going backwards).
            checkpoints = sorted(checkpoints, key=lambda c: c.metadata["checkpoint_ind"], reverse=True)
            checkpoint_vals = [ checkpoint.metadata["checkpoint_relevance" ] for checkpoint in checkpoints ]

            if not DEBUG:
                Promise.clear_all()

            return checkpoint_vals, out_adj_list, topo_exec_order, fcn_map