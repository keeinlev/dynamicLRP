import torch
import time
from enum import Enum
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
from unbind_backward_promise import UnbindBackwardPromise
from sum_backward_promise import SumBackwardPromise
from cat_backward_promise import CatBackwardPromise
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

class RunMode(Enum):
    NORMAL = 0
    PROMISE_TRAVERSAL = 1
    PROMISE_FULFILLMENT = 2

    @property
    def is_normal(self):
        return self == RunMode.NORMAL

    @property
    def is_traversal(self):
        return self == RunMode.PROMISE_TRAVERSAL

    @property
    def is_fulfillment(self):
        return self == RunMode.PROMISE_FULFILLMENT


def group_and_categorize_inputs(input_tracker_list: list[tuple[Union[Promise, torch.Tensor], int]], run_mode: RunMode):
    """Groups all Node relevance inputs by their index that corresponds them to one of the Node's forward outputs,
        and then further groups them by what category of relevance input they are, complete promise, pending promise,
        and tensor.
    input_tracker_list contains all relevance inputs from forward consumer Nodes, and the index of the
        forward output that they consume.
    Outputs the inputs to the Node's propagation function, grouped as described above, in the following structure:
        [ (idx, (complete_promises_list, pending_promises_list, tensors_list)), ... ]"""
    idx_groups = {}
    for input_, input_idx in input_tracker_list:
        if input_idx not in idx_groups:
            idx_groups[input_idx] = ([], [], [])
        complete_promises, pending_promises, tensors = idx_groups[input_idx]
        if isinstance(input_, torch.Tensor):
            tensors.append(input_)
        elif isinstance(input_, Promise):
            if input_.complete:
                complete_promises.append(input_)
            else:
                pending_promises.append(input_)
        elif input_ == 0.0:
            continue
        elif input_ is None and run_mode.is_traversal:
            continue
        else:
            raise ValueError(input_)
    return list(idx_groups.items())
    
def node_preprocess(curnode: Node, metadata):
    """Use this to hook in any kind of extra metadata specific to a certain kind of Node before processing."""
    node_name = type(curnode).__name__
    if node_name == "AddBackward0":
        curnode.metadata["promise_type"] = AddBackwardPromise
    elif node_name == "UnbindBackward0":
        curnode.metadata["promise_type"] = UnbindBackwardPromise
    elif node_name == "SumBackward0":
        curnode.metadata["promise_type"] = SumBackwardPromise
    elif node_name == "CatBackward0":
        curnode.metadata["promise_type"] = CatBackwardPromise
    elif node_name == "DecomposedConvolutionBackward0":
        curnode.metadata["conv_layer"] = metadata.get("conv_layer", 0)
        curnode.metadata["use_gamma"] = metadata.get("use_gamma", False)
        metadata["conv_layer"] += 1
    elif node_name == "MmBackward0":
        curnode.metadata["mm_layer"] = metadata.get("mm_layer", 0)
        curnode.metadata["use_gamma"] = metadata.get("use_gamma", False)
        metadata["mm_layer"] += 1

def lrp_engine(
        hidden_states : torch.Tensor,
        out_adj_list: Union[set[Node], set[int]] = None, # Reusable
        topo_exec_order : list[int] = None, # Reusable
        fcn_map : dict[str, Callable] = None, # Reusable
        params_to_interpret : list[torch.Tensor] = None, # Set at first LRP call if you want weight attributions
        param_node_inds : list[int] = None, # Reusable
        no_recompile = False,
        use_gamma = False,
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

        mm_counter = 0
        conv_layer_counter = 0

        if out_adj_list is None or topo_exec_order is None or fcn_map is None:
            in_adj_list, out_adj_list, names, ind_to_node, num_nodes, updated_root, param_nodes = make_graph(hidden_states, params_to_interpret, return_topo_dict=True)

            root = hidden_states.grad_fn
            if updated_root is not None:
                root = updated_root[0]

            Promise.ind_to_node = ind_to_node

            input_tracker : dict[Node, list] = { k : [] for k in list(in_adj_list.keys()) }
            checkpoints = list(filter(lambda k: type(k).__name__ == "LRPCheckpointBackward", list(in_adj_list.keys())))
            num_params_reached = 0
            num_checkpoints_reached = 0

            fcn_map = LRPPropFunctions.generate_prop_fcn_map(names)

            visited = set()

            # Setup the first iteration
            input_tracker[root] = [ (relevance, 0) ]
            stack : list[Node] = [root]
            in_adj_list[root] = []
            nodes_pending = { k : len(v) for k, v in list(in_adj_list.items()) }

            promise_queue : list[Node] = []

            promise_traversal_stack : list[Node] = []

            RUN_MODE = RunMode.NORMAL

            while (stack or promise_traversal_stack or promise_queue) and (num_checkpoints_reached < len(checkpoints) or num_params_reached < len(param_nodes)):
                
                curnode = None

                ####### RUN MODE DETERMINATION
                
                # Decide where we should take curnode from
                if promise_queue and any(fcn.metadata["promise"]["complete"] for fcn in promise_queue if "promise" in fcn.metadata):
                    # Search for the first complete promise in the queue.
                    curnode = next(( fcn for fcn in promise_queue if fcn.metadata["promise"]["complete"] ))
                    idx = promise_queue.index(curnode)
                    promise_queue = promise_queue[:idx] + promise_queue[idx + 1:]
                    RUN_MODE = RunMode.PROMISE_FULFILLMENT

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
                    RUN_MODE = RunMode.NORMAL

                    # It's possible that this Node was already waiting on a Promise too, in the diamond deadlock case
                    # If so, just remove the other link to the same promise for cleanliness
                    if "promise" in curnode.metadata:
                        # assert curnode.metadata["promise"] is curnode.metadata["pre_promise"].promise, \
                        #     f"Node has both promise and pre_promise metadata fields, but the promises are not equal, traversal logic error. {curnode.metadata['promise']}, {curnode.metadata['pre_promise'].promise}"
                        del curnode.metadata["promise"]

                elif promise_traversal_stack:
                    # Second priority is promise traversal, which overrides the requirement for all inputs to land
                    # before traversing a node. However, the promise will not have its rins computed until 
                    curnode = promise_traversal_stack[0]
                    promise_traversal_stack = promise_traversal_stack[1:]
                    RUN_MODE = RunMode.PROMISE_TRAVERSAL

                elif stack:
                    # Fallback to main stack
                    curnode = stack[0]
                    stack = stack[1:]
                    RUN_MODE = RunMode.NORMAL
                else:
                    raise RuntimeError("No valid curnode candidate was found.")

                ####### END RUN MODE DETERMINATION

                if curnode not in visited:
                    visited.add(curnode)
                    metadata = {
                        "conv_layer": conv_layer_counter,
                        "mm_layer": mm_counter,
                        "use_gamma": use_gamma,
                    }
                    node_preprocess(curnode, metadata)
                    conv_layer_counter = metadata["conv_layer"]
                    mm_counter = metadata["mm_layer"]

                    # Check and set input node
                    if "save_relevance" in curnode.metadata:
                        if type(curnode).__name__ != "EmbeddingBackward0":
                            curnode.metadata["relevance"] = input_tracker[curnode][0][0] # TODO: Come back for the Promise case

                ####### INPUT MERGING/HANDLING
                sorted_and_merged_inputs = {}
                if not RUN_MODE.is_fulfillment:
                    curnode_inputs = input_tracker[curnode]

                    # Group all inputs by their indices then categorize into either pending promises, complete promises, or tensors
                    try:
                        categorized_inputs = group_and_categorize_inputs(curnode_inputs, RUN_MODE)
                    except ValueError as e:
                        return curnode, curnode_inputs, in_adj_list, out_adj_list, ValueError(f"Expected relevance input to Node {curnode} to be type Promise or Tensor, but got {type(e.args[0])} instead.")
                    
                    for idx, (complete_promise_inputs, pending_promise_inputs, tensor_inputs) in categorized_inputs:
                        if not complete_promise_inputs and not pending_promise_inputs and not tensor_inputs:
                            continue

                        if RUN_MODE.is_traversal and not pending_promise_inputs:
                            # If in PTM, we only care about the lone pending Promise input. Everything else we will
                            # assign and connect when we come back to this Node later on.
                            sorted_and_merged_inputs[idx] = None
                            continue
                        
                        # 3 more cases: NTM + pending Promises, NTM no pending Promises, PTM + pending Promises
                        # each NTM case breaks down into 2 sub-cases, revisiting Pre-Promise or not.

                        # Aggregate all inputs into one Tensor or Promise
                        curnode_in_rel = sum(tensor_inputs)

                        ####### PRE-PROMISE RETRIEVAL
                        # Consider that we traverse the same Node at most twice at this stage. Once possibly for PTM, and the second
                        # in Normal Traversal Mode. This is because a Node will not be placed back in PTM if it already
                        # has set the "pre_promise" key in its metadata, i.e. if we've already processed it in PTM.
                        if "pre_promise" in curnode.metadata:
                            # This will handle both NTM revisiting Pre-Promise sub cases
                            # Essentially checking if this is a Node we traversed in PTM and are now traversing again in Normal Mode.
                            pre_promise : Promise = curnode.metadata["pre_promise"]
                            assert pre_promise.ready, "Pre-Promise assumed to be ready upon re-traversal but was not."
                            assert nodes_pending[curnode] == 0, "Re-traversing Node with Pre-Promise but not all of its inputs have landed."

                            if not pre_promise.complete:
                                # Should always be True on the first time back around!!!
                                # Accumulate the landed Tensor relevance
                                if isinstance(curnode_in_rel, torch.Tensor):
                                    pre_promise.accumulate_rout(curnode_in_rel, idx)
                                
                                for parent in pre_promise.parents:
                                    if pre_promise not in parent.children:
                                        parent.children.append(pre_promise)
                                        # parent.promise["tail_nodes"].union(pre_promise["tail_nodes"])
                                        if parent.complete:
                                            pre_promise.pending_parents -= 1
                                            pre_promise.accumulate_rout(parent.rin, idx)

                                for pending_promise in pending_promise_inputs:
                                    if pending_promise not in pre_promise.parents:
                                        # Create parent-child link now that we know all inputs have landed.
                                        # The Pre-Promise becomes the Aggregate Promise, except it has already been fully built.
                                        pre_promise.parents.append(pending_promise)
                                        pre_promise.pending_parents += 1
                                        pre_promise.parent_idxs[pending_promise] = idx
                                    if pre_promise not in pending_promise.children:
                                        pending_promise.children.append(pre_promise)
                                    pending_promise.arg_node_ind = curnode.metadata["topo_ind"]

                            else:
                                # This actually should never happen, so raise an Error
                                raise RuntimeError("Pre-Promise was found completed at the second traversal of the Node. Contradicts graph traversal heuristic or Node is being traversed too many times.")
                        ####### END PRE-PROMISE RETRIEVAL


                        elif pending_promise_inputs:
                            # PTM and NTM + pending Promises case on first traversal of Node

                            # If PTM, consider that there is only one Promise input across all indices, and this will create the Pre-Promise.
                            # In promise traversal mode this will be True
                            agg_promises = compound_promises(pending_promise_inputs, curnode.metadata["topo_ind"], RUN_MODE.is_traversal, RUN_MODE.is_traversal)
                            if tensor_inputs and not RUN_MODE.is_traversal:
                                if len(pending_promise_inputs) == 1:
                                    # In this case, compound_promises did not return a new DummyPromise, since there was only one Promise to compound
                                    # However, we now know there are also Tensor relevance inputs from other in-edges, which we cannot account for
                                    # in the input-agnostic run if we merge them into the Promise chain at this point. It needs to be at the start of
                                    # a Promise.
                                    agg_promises = compound_promises(pending_promise_inputs, curnode.metadata["topo_ind"], single_promise_override=True)
                                # We don't add the tensor relevances to new Pre-Promises since we will do that anyway on the revisit of the Node
                                agg_promises.set_rout(curnode_in_rel)
                            curnode_in_rel = agg_promises
                            if RUN_MODE.is_traversal:
                                sorted_and_merged_inputs[idx] = curnode_in_rel
                                break

                        else:
                            # NTM and no pending Promises, best case
                            curnode_in_rel += sum([ p.rin for p in complete_promise_inputs ])

                        sorted_and_merged_inputs[idx] = curnode_in_rel
                    
                    if "pre_promise" in curnode.metadata:
                        try:
                            pre_promise.trigger_promise_completion()
                        except RuntimeError as e:
                            return curnode, pre_promise, in_adj_list, out_adj_list, e


                        if pre_promise.complete:
                            # Put the tail nodes on the stack if they are ready
                            tail_nodes = list(pre_promise.promise["tail_nodes"])
                            
                            if len(tail_nodes) == 1 and tail_nodes[0] == curnode:
                                sorted_and_merged_inputs[0] = pre_promise.rin
                            else:
                                ready_tail_nodes = []
                                for tail_node in tail_nodes:
                                    if nodes_pending[tail_node] == 0 and tail_node not in stack and tail_node not in promise_queue:
                                        ready_tail_nodes.append(tail_node)
                                # There is a case where the start node is a tail node, if one of the children was None, see in the child input propagation step.
                                # So we take the start node out of ready_tail_nodes, to avoid retraversing this Node.
                                ready_tail_nodes = [ rtn for rtn in ready_tail_nodes if rtn != curnode ]
                                stack = ready_tail_nodes + stack
                                continue
                            # promise_queue.append(curnode)

                        # Consider that at any tail node of a Pre-Promise, the arg was set but the Promise did not complete,
                        # so the Node gets placed on the promise queue. If our Pre-Promise has now completed as a result of
                        # connecting its delayed parents and retriggering completion, then all those Nodes on the promise
                        # queue should now be ready to be dequeued in following iterations, therefore we have nothing left
                        # to do at this Node.
                        # If the Pre-Promise is still not complete, that means our backward prop is stuck at an earlier parent,
                        # who itself has to wait for its other branches to resolve. But when they do, the parent will do our
                        # job of going back down the Promise tree and retriggering this Pre-Promise, until its tail nodes are
                        # reached. Therefore, in both cases, we simply continue from this Node.
                    ####### END INPUT MERGING

                else:
                    # In promise fulfillment mode, use the completed promise's rin for traversing curnode.
                    sorted_and_merged_inputs[0] = curnode.metadata["promise"]["rins"][curnode.metadata["promise_idx"]]


                if RUN_MODE.is_traversal:
                    # We want to save this so later we'll know we've already traversed this node.
                    curnode.metadata["pre_promise"] = curnode_in_rel

                # Convert to a tuple, filling in missing inputs if needed.
                if not sorted_and_merged_inputs:
                    continue
                num_inputs = max(list(sorted_and_merged_inputs.keys())) + 1
                finalized_inputs = tuple([ sorted_and_merged_inputs[i] if i in sorted_and_merged_inputs else None for i in range(num_inputs)])
                if len(finalized_inputs) == 1:
                    finalized_inputs = finalized_inputs[0]

                ####### PROPAGATION FCN AND PROMISE QUEUE HANDLING

                # Call the propagation function for the node
                try:
                    curnode_outputs = fcn_map[type(curnode).__name__](curnode, finalized_inputs)
                except Exception as e:
                    print(e)
                    return curnode, curnode_inputs, in_adj_list, out_adj_list, e

                if isinstance(curnode_outputs, Promise) and curnode_outputs.arg is not None and not curnode_outputs.complete:
                    # Node is waiting on Promise to be completed, add to promise queue and come back later.
                    if curnode not in promise_queue:
                        curnode.metadata["promise"] = curnode_outputs.promise
                        curnode.metadata["promise_idx"] = getattr(curnode_outputs, "idx", 0)
                        promise_queue.append(curnode)
                    continue

                if RUN_MODE.is_fulfillment:
                    if not DEBUG:
                        Promise.clear_args_and_rout_raw(curnode.metadata["promise"])
                    del curnode.metadata["promise"], curnode.metadata["promise_idx"]

                if not RUN_MODE.is_traversal and not DEBUG:
                    # We can free up some memory because we will no longer need to access these inputs
                    # Chained promises will maintain their relationships via their class instance members.
                    if curnode in input_tracker and nodes_pending[curnode] == 0 and curnode not in promise_queue:
                        del input_tracker[curnode]

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

                # According to next_functions
                children = out_adj_list[curnode]

                try:
                    # Children may contain None, like grad_fn.next_functions, to keep integrity of input tracking
                    if len(children) == 0 or all(child is None for child in children):
                        continue
                    elif len(children) == 1:
                        curnode_outputs = [ curnode_outputs ]
                        
                    elif len(children) != len(curnode_outputs):
                        raise ValueError(f"Mismatch: {len(children)} children but {len(curnode_outputs)} outputs from {curnode}.")
                except TypeError as e:
                    print(curnode, children, curnode_outputs)
                    raise e

                # Update child inputs
                for i, (child, input_idx) in enumerate(children):
                    if child is None:
                        # Discard the input (it shouldn't have value anyway), if it's a promise make it a zero-promise
                        if isinstance(curnode_outputs[i], Promise):
                            curnode_outputs[i].setarg(0.0, curnode, lambda node: 0.0)
                        continue
                    input_tracker[child].append((curnode_outputs[i], input_idx))
                    nodes_pending[child] -= 1
                    try:
                        assert nodes_pending[child] >= 0, f"Negative pending count for node {child} while running in mode {RUN_MODE}"
                        assert len(input_tracker[child]) <= len(in_adj_list[child]), \
                            f"Too many inputs landed for {child}"
                    except AssertionError as e:
                        return curnode, in_adj_list, out_adj_list, input_tracker, e

                ####### END OUTPUT PROPAGATION TO CHILDREN


                ####### SORTING CHILDREN TO WHICH STACK THEY SHOULD GO TO

                # Collect children who now have all their inputs or that have promise(s) depending on them.
                ready_children : list[Node] = [] # All children who have all their inputs landed
                promise_depends_on : list[Node] = [] # All children who do not have all their inputs landed but have at least one incomplete promise input landed.
                for i, (child, input_idx) in enumerate(children):
                    if child is None:
                        continue
                    if nodes_pending[child] == 0 and child not in promise_queue:
                        ready_children.append(child)
                    elif isinstance(curnode_outputs[i], Promise) and not curnode_outputs[i].complete and "pre_promise" not in child.metadata:
                        promise_depends_on.append(child)

                promise_traversal_stack = promise_depends_on + promise_traversal_stack
                stack = ready_children + stack
                num_checkpoints_reached = sum([ "relevance" in checkpoint.metadata for checkpoint in checkpoints])
                num_params_reached = sum([ "relevance" in param_node.metadata for param_node in param_nodes ])

                ####### END SORTING CHILDREN TO WHICH STACK THEY SHOULD GO TO

            end_time = time.time()
            if DEBUG:
                print(f"propagation took {end_time - start_time} seconds")

                # Checking conservation holds across the entire propagation
                # The frontier includes:
                # a) true leaf nodes (no children)
                # b) nodes which received inputs but were never traversed due to computation ending early

                # Note: this must be run in Debug mode or else there will be relevance that gets lost from clearing of cached inputs/Promise values.

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
                        if isinstance(input_, Promise):
                            if input_.complete:
                                total_in += input_.rin.sum()
                            else:
                                continue
                        elif isinstance(input_, torch.Tensor):
                            total_in += input_.sum()
                        elif isinstance(input_, float):
                            total_in += input_
                    total_frontier_in += total_in
                print("TOTAL FRONTIER RELEVANCE AFTER PROPAGATION: ", total_frontier_in)

            try:
                # Checkpoints sorted in desc because they are indexed in the order that we save them in (going backwards).
                checkpoints = sorted(checkpoints, key=lambda c: c.metadata["checkpoint_ind"], reverse=True)
                checkpoint_vals = [ checkpoint.metadata["relevance" ] for checkpoint in checkpoints ]
                param_node_inds = []
                param_node_vals = []
                for node in param_nodes:
                    param_node_inds.append(node.metadata["topo_ind"])
                    param_node_vals.append(node.metadata["relevance"])
            except KeyError as e:
                print(f"Some checkpoints were not reached during traversal, see metadata objects: {[ node.metadata for node in checkpoints ]}")
                return curnode, checkpoints, in_adj_list, out_adj_list, e

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
                param_node_inds
            )

            if not DEBUG:
                Promise.clear_all()
            
            Promise.repair_all_parent_child_connections()

            return (checkpoint_vals, param_node_vals), out_adj_list, topo_exec_order, fcn_map, param_node_inds
        else:
            # Run execution plan made from first pass
            # All adjacency lists and node orders are in terms of topological indices now

            # Need to re-map the graph to all the indices based on topo sort
            _, _, _, ind_to_node, _, _, _ = make_graph(hidden_states, return_topo_dict=True)

            Promise.ind_to_node = ind_to_node

            checkpoints = list(filter(lambda node: type(node).__name__ == "LRPCheckpointBackward", list(ind_to_node.values())))

            input_frontier : dict[int, list[torch.Tensor]] = { ind : {0: 0.0} for ind in topo_exec_order }

            # Setup first iteration
            input_frontier[hidden_states.grad_fn.metadata["topo_ind"]] = {0: relevance}

            # First pre-process all Promises by updating the starting fwd_shapes and setting every leaf Promise arg
            Promise.update_all_starting_fwd_shapes()
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

                if type(node).__name__ == "DecomposedConvolutionBackward0":
                    node.metadata["conv_layer"] = conv_layer_counter
                    node.metadata["use_gamma"] = use_gamma
                    conv_layer_counter += 1

                if type(node).__name__ == "MmBackward0":
                    node.metadata["mm_layer"] = mm_counter
                    node.metadata["use_gamma"] = use_gamma
                    mm_counter += 1

                rel = tuple([ elem if i in input_frontier[node_ind] else None for i, elem in list(input_frontier[node_ind].items()) ])
                # if len(rel) == 1:
                #     rel = rel[0]

                if (promise := Promise.start_nodes_to_promise.get(node_ind)) and isinstance(promise, Promise) and promise not in fulfilled_promises and \
                        promise.start_ind != promise.arg_node_ind:
                    if not promise.ready:
                        print(node_ind, promise.start_ind, promise.arg_node_ind)
                    assert promise.ready, f"During running of execution plan, promise was missing arg(s) {promise.promise}"

                    # Prep the promise
                    for idx, r in enumerate(rel):
                        if not isinstance(r, float):
                            try:
                                promise.accumulate_rout(r, idx)
                            except RuntimeError as e:
                                print(promise, promise.rout.shape, r.shape, rel)
                                raise e
                    if isinstance(promise.rout, float):
                        print(promise.promise, promise.start_ind)
                        raise TypeError
                    promise.compute_rins()

                    # Do the same for other branches if applicable
                    curr_branch = promise
                    is_start = True
                    while curr_branch != promise or is_start:
                        if curr_branch in fulfilled_promises or curr_branch.arg_node_ind not in input_frontier:
                            curr_branch = curr_branch.other_branch
                            continue
                        res2 = curr_branch.bwd(curr_branch.rin)
                        rin_landing_ind = curr_branch.arg_node_input_index(out_adj_list)
                        input_frontier[curr_branch.arg_node_ind][rin_landing_ind] += res2
                        if (child_promise := Promise.start_nodes_to_promise.get(curr_branch.arg_node_ind)) and isinstance(child_promise, Promise):
                            child_promise.pending_parents -= 1
                        fulfilled_promises.add(curr_branch)

                        if curr_branch.other_branch is None:
                            break
                        curr_branch = curr_branch.other_branch
                        is_start = False
                    
                    promise.set_complete()
                else:
                    if len(rel) == 1:
                        rel = rel[0]
                    try:
                        res = fcn_map[type(node).__name__](node, rel)
                    except (AttributeError, RuntimeError) as e:
                        print(node_ind, [ idx for idx in topo_exec_order if node_ind in out_adj_list[idx] ], input_frontier[node_ind])
                        raise e
                    # Distribute the relevance the same way as first pass
                    if not isinstance(res, tuple):
                        res = [ res ]

                    for i, (child, idx) in enumerate(out_adj_list[node_ind]):
                        if child is None or child not in input_frontier or (isinstance(res[i], float) and res[i] == 0.0):
                            continue

                        if idx not in input_frontier[child]:
                            input_frontier[child][idx] = 0.0
                        input_frontier[child][idx] += res[i]
                
                if "relevance" not in node.metadata and node_ind in param_node_inds:
                    node.metadata["relevance"] = input_frontier[node_ind][0]

                # Move the frontier forward
                # Since we are iterating over a topological ordering, when we are a certain node, we are certain
                # that all of its dependencies have been visited at that point, so we can safely delete its inputs.

                del input_frontier[node_ind]

                iter_time = time.time() - iter_start_time
                if iter_time > 1.5 / len(topo_exec_order) and DEBUG:
                    print(f"Node {node_ind}, {node} took {iter_time}s")

            end_time = time.time()
            if DEBUG:
                print(f"propagation took {end_time - start_time} seconds")

            # Checkpoints sorted in desc because they are indexed in the order that we save them in (going backwards).
            checkpoints = sorted(checkpoints, key=lambda c: c.metadata["checkpoint_ind"], reverse=True)
            checkpoint_vals = [ checkpoint.metadata["relevance" ] for checkpoint in checkpoints ]

            param_node_vals = [ ind_to_node[node_ind].metadata["relevance"] for node_ind in param_node_inds ]

            if not DEBUG:
                Promise.clear_all()

            return (checkpoint_vals, param_node_vals), out_adj_list, topo_exec_order, fcn_map, param_node_inds