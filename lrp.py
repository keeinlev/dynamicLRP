import torch
from torch.autograd.graph import Node
from lrp_graph import make_graph
from lrp_prop_fcns import LRPPropFunctions
from add_backward_promise import AddBackwardPromise, compound_promises
from util import create_checkpoint
from transformers import AutoModel, AutoTokenizer

def checkpoint_hook(module, input, output):
    return create_checkpoint(output)

model_name = "zhihan1996/DNABERT-2-117M"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModel.from_pretrained(model_name, trust_remote_code=True)

for layer_module in model.encoder.layer:
    layer_module.attention.self.register_forward_hook(checkpoint_hook)

dna = "ACGTAGCATCGGATCTATCTATCGACACTTGGTTATCGATCTACGAGCATCTCGTTAGC"
inputs = tokenizer(dna, return_tensors = 'pt')["input_ids"]
hidden_states : torch.Tensor = model(inputs, requires_grad=True)[0] # [1, sequence_length, 768]

in_adj_list, out_adj_list, names = make_graph(hidden_states)
input_tracker : dict[Node, list] = { k : [] for k in list(in_adj_list.keys()) }
checkpoints = list(filter(lambda k: type(k).__name__ == "LRPCheckpointBackward", list(in_adj_list.keys())))
num_checkpoints_reached = 0

fcn_map = LRPPropFunctions.generate_prop_fcn_map(names)

# visited1 = set()
with torch.no_grad():
    # Create the first relevance layer via max logit.
    m = hidden_states.max(-1)
    relevance = torch.zeros_like(hidden_states)
    b, s, d = hidden_states.shape
    for i, inds in enumerate(m.indices):
        relevance[i,list(range(s)),inds] = torch.ones_like(m.values[0])

    # Setup the first iteration
    input_tracker[hidden_states.grad_fn] = [ relevance ]
    stack = [hidden_states.grad_fn]
    in_adj_list[hidden_states.grad_fn] = []
    nodes_pending = { k : len(v) for k, v in list(in_adj_list.items()) }

    promise_queue : list[Node] = []

    promise_traversal_stack = []
    promise_traversal_mode = False

    promise_fulfillment_mode = False

    while (stack or promise_traversal_stack or promise_queue) and num_checkpoints_reached < len(checkpoints):
        
        curnode = None
        
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

        curnode_inputs = input_tracker[curnode]

        # visited1.add(curnode) # For debugging

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
                elif isinstance(input_, AddBackwardPromise) and input_.complete:
                    complete_promise_inputs.append(input_)
                elif isinstance(input_, AddBackwardPromise):
                    pending_promise_inputs.append(input_)
                elif input_ == 0.0:
                    continue
                else:
                    print(input_)
                    raise ValueError(f"Expected relevance input to Node {curnode} to be type AddBackwardPromise or Tensor, but got {type(input_)} instead.")
    
            if not complete_promise_inputs and not pending_promise_inputs and not tensor_inputs:
                continue
    
            # Aggregate all inputs into one Tensor or AddBackwardPromise
            curnode_in_rel = sum(tensor_inputs) + sum([ p.rin for p in complete_promise_inputs ])
            if pending_promise_inputs:
                # In promise traversal mode this will be True
                agg_promises = compound_promises(pending_promise_inputs, promise_traversal_mode, promise_traversal_mode)
                if curnode_in_rel != 0:
                    curnode_in_rel = agg_promises + curnode_in_rel
                else:
                    curnode_in_rel = agg_promises
        else:
            curnode_in_rel = curnode.metadata["promise"]["rins"][curnode.metadata["promise_idx"]]


        if not promise_traversal_mode and "pre_promise" in curnode.metadata:
            # We have already traversed a promise tree, but have not calculated its bwd,
            # since it was done in promise traversal mode.
            pre_promise : AddBackwardPromise = curnode.metadata["pre_promise"]

            assert pre_promise.ready, f"Pre-promise at {curnode} was assumed to be ready but was not."

            if not pre_promise.complete:
                if isinstance(curnode_in_rel, AddBackwardPromise):
                    # If there is still pending promises at this node, try to complete them via the aggregate promise.
                    # In the case this completes and propagates relevance down, we will have to pick up from the tail nodes
                    # of the aggregate promise.
                    curnode_in_rel.children.append(pre_promise)
                    curnode_in_rel.setarg(pre_promise.arg1 + pre_promise.arg2)
                else:
                    pre_promise.accumulate_rout(curnode_in_rel)
                    pre_promise.trigger_promise_completion()

            tail_nodes = pre_promise.promise["tail_nodes"]
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

        if promise_traversal_mode:
            # We want to save this so later we'll know we've already traversed this node.
            curnode.metadata["pre_promise"] = curnode_in_rel

        # Call the propagation function for the node
        curnode_outputs = fcn_map[type(curnode).__name__](curnode, curnode_in_rel)

        if isinstance(curnode_outputs, AddBackwardPromise) and curnode_outputs.arg is not None and not curnode_outputs.complete:
            # Node is waiting on Promise to be completed, add to promise queue and come back later.
            curnode.metadata["promise"] = curnode_outputs.promise
            curnode.metadata["promise_idx"] = curnode_outputs.idx
            promise_queue.append(curnode)
            continue

        # Children may contain None, like grad_fn.next_functions, to keep integrity of input tracking
        if len(children) == 0 or all(child is None for child in children):
            continue
        elif len(children) == 1:
            # if isinstance(curnode_outputs, tuple):
            #     curnode_outputs = [ curnode_outputs[0] ]
            # else:
            curnode_outputs = [ curnode_outputs ]
            
        elif len(children) != len(curnode_outputs):
            raise ValueError(f"Mismatch: {len(children)} children but {len(curnode_outputs)} outputs from {curnode}.")


        # Update child inputs
        for i, child in enumerate(children):
            if child is None:
                # Discard the input (it shouldn't have value anyway), if it's a promise make it a zero-promise
                if isinstance(curnode_outputs[i], AddBackwardPromise):
                    # Manually set the arg to not trigger any additional side effects
                    curnode_outputs[i].promise["args"][curnode_outputs[i].idx] = 0.0
                nodes_pending[child] -= 1 # Still need to do this to keep the traversal working
                continue
            input_tracker[child].append(curnode_outputs[i])
            nodes_pending[child] -= 1
            assert nodes_pending[child] >= 0, f"Negative pending count for node {child}"
            assert len(input_tracker[child]) <= len(in_adj_list[child]), \
                f"Too many inputs landed for {child}"

        # Collect children who now have all their inputs or that have promise(s) depending on them.
        ready_children = []
        promise_depends_on = []
        for i, child in enumerate(children):
            if child is None:
                continue
            if nodes_pending[child] == 0 and child not in promise_queue:
                ready_children.append(child)
            elif isinstance(curnode_outputs[i], AddBackwardPromise) and not curnode_outputs[i].complete and "pre_promise" not in child.metadata:
                promise_depends_on.append(child)

        promise_traversal_stack = promise_depends_on + promise_traversal_stack
        stack = ready_children + stack
        num_checkpoints_reached = sum([ "checkpoint_relevance" in checkpoint.metadata for checkpoint in checkpoints])


checkpoint_vals = [ checkpoint.metadata["checkpoint_relevance" ] for checkpoint in checkpoints ]
