import torch

DISABLED_HEAD_EVAL = False
NUM_HEADS = 12
HEAD_DIM = 64

def make_hook(disabled_head_ind):
    def disable_head_hook(module: torch.nn.Module, input: torch.Tensor, output: torch.Tensor):
        """
        Zeroes out the section of attention output that corresponds to DISABLED_HEAD_IND.
        Meant to be run only when in inference after training is done.
        """
        if module.training:
            return output

        start = disabled_head_ind * HEAD_DIM
        end = start + HEAD_DIM

        output = output.clone()
        output[:,start:end].zero_()

        return output

    return disable_head_hook