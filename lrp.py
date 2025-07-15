from transformers import AutoModel, AutoTokenizer
import torch
from torch.autograd.graph import Node
from lrp_graph import make_graph
from lrp_prop_fcns import LRPPropFunctions

model_name = "zhihan1996/DNABERT-2-117M"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModel.from_pretrained(model_name, trust_remote_code=True)

dna = "ACGTAGCATCGGATCTATCTATCGACACTTGGTTATCGATCTACGAGCATCTCGTTAGC"
inputs = tokenizer(dna, return_tensors = 'pt')["input_ids"]
hidden_states : torch.Tensor = model(inputs, requires_grad=True)[0] # [1, sequence_length, 768]

in_adj_list, out_adj_list, names = make_graph(hidden_states)
input_tracker : dict[Node, list[None | torch.Tensor]] = { k : [ None for _ in range(len(v)) ] for k, v in list(in_adj_list.items()) }

fcn_map = LRPPropFunctions.generate_prop_fcn_map(names)

with torch.no_grad():
    # Create the first relevance layer via max logit.
    m = hidden_states.max(-1)
    relevance = torch.zeros_like(hidden_states)
    b, s, d = hidden_states.shape
    for i, inds in enumerate(m.indices):
        relevance[i:list(range(s)):inds] = torch.ones_like(m.values[0])

    # Setup the first iteration
    input_tracker[hidden_states.grad_fn] = [ relevance ]
    stack = [hidden_states.grad_fn]

    while stack:
        # Pop first element
        curnode = stack[0]
        stack = stack[1:]

        curnode_inputs = input_tracker[curnode]
        if any([ x is None for x in curnode_inputs ]):
            # Node hasn't received all its relevance yet, push it to the back.
            stack = stack + [curnode]
            continue

        curnode_in_rel = sum(curnode_inputs, torch.zeros_like(curnode_inputs[0]))
        # Call the propagation function for the node
        curnode_outputs = fcn_map[curnode.name](curnode, curnode_in_rel)

        if curnode_outputs is None:
            # Node is waiting on Promise to be completed, come back later.
            stack = stack + [curnode]
            continue

        children = out_adj_list[curnode]
        if len(children) == 1:
            input_tracker[curnode][0] = curnode_outputs
        else:
            input_tracker[curnode] = curnode_outputs

        # Iterate DFS-style
        stack = children + curnode


checkpoints = list(filter(lambda k: type(k).__name__ == "LRPCheckpointBackward", list(in_adj_list.keys())))
checkpoint_vals = [ checkpoint.metadata["checkpoint_relevance"] for checkpoint in checkpoints ]