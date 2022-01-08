
# 
import torch
import math
import numpy as np
import json

from typing import Tuple, List


def tensor2np(x):
    """Convert torch.Tensor to np.ndarray
    Args:
        x (torch.Tensor)
    Returns:
        np.ndarray
    """
    if x is None:
        return x 
    return x.cpu().detach().numpy()

def log_add(args: List[int]) -> float:
    """
    Stable log add 
    """
    if all(a == -float('inf') for a in args):
        return -float('inf')
    a_max = max(args)
    lsp = math.log(sum(math.exp(a - a_max) for a in args))
    return a_max + lsp


def remove_duplicates_and_blank(hyp: List[int]) -> List[int]:
    """
    remove duplicates and blank 
    """
    new_hyp: List[int] = []
    cur = 0
    while cur < len(hyp):
        if hyp[cur] != 0:
            new_hyp.append(hyp[cur])
        prev = cur 
        while cur < len(hyp) and hyp[cur] == hyp[prev]:
            cur += 1
    
    return new_hyp

def make_pad_mask(lengths: torch.Tensor, le : bool = True) -> torch.Tensor:
    """Make mask tensor containing indices of padded part.

    See description of make_non_pad_mask.

    Args:
        lengths (torch.Tensor): Batch of lengths (B,).
    Returns:
        torch.Tensor: Mask tensor containing indices of padded part.

    Examples:
        >>> lengths = [5, 3, 2]
        >>> make_pad_mask(lengths)
        # masks = [[0, 0, 0, 0 ,0],
        #          [0, 0, 0, 1, 1],
        #          [0, 0, 1, 1, 1]]
        masks = [[1, 1, 1, 1, 1],
                 [1, 1, 1, 0, 0],
                 [1, 1, 0, 0, 0]]
    """
    batch_size = int(lengths.size(0))
    max_len = int(lengths.max().item())
    seq_range = torch.arange(0,
                             max_len,
                             dtype=torch.int64,
                             device=lengths.device)
    seq_range_expand = seq_range.unsqueeze(0).expand(batch_size, max_len)
    seq_length_expand = lengths.unsqueeze(-1)
    # mask = seq_range_expand >= seq_length_expand
    # fix: torch.float32 -> torch.int32
    if le:
        mask = (seq_range_expand < seq_length_expand).type(torch.int32)
    else:
        mask = (seq_range_expand >= seq_length_expand).type(torch.int32)

    # print(mask)
    return mask

class GlobalCMVN(torch.nn.Module):
    def __init__(self,
                 mean: torch.Tensor,
                 istd: torch.Tensor,
                 norm_var: bool = True):
        """
        Args:
            mean (torch.Tensor): mean stats
            istd (torch.Tensor): inverse std, std which is 1.0 / std
        """
        super().__init__()
        assert mean.shape == istd.shape
        self.norm_var = norm_var
        # The buffer can be accessed from this module using self.mean
        self.register_buffer("mean", mean)
        self.register_buffer("istd", istd)

    def forward(self, x: torch.Tensor):
        """
        Args:
            x (torch.Tensor): (batch, max_len, feat_dim)

        Returns:
            (torch.Tensor): normalized feature
        """
        x = x - self.mean
        if self.norm_var:
            x = x * self.istd
        return x

def _load_json_cmvn(json_cmvn_file):
    """ Load the json format cmvn stats file and calculate cmvn

    Args:
        json_cmvn_file: cmvn stats file in json format

    Returns:
        a numpy array of [means, vars]
    """
    with open(json_cmvn_file) as f:
        cmvn_stats = json.load(f)

    means = cmvn_stats['mean_stat']
    variance = cmvn_stats['var_stat']
    count = cmvn_stats['frame_num']
    for i in range(len(means)):
        means[i] /= count
        variance[i] = variance[i] / count - means[i] * means[i]
        if variance[i] < 1.0e-20:
            variance[i] = 1.0e-20
        variance[i] = 1.0 / math.sqrt(variance[i])
    cmvn = np.array([means, variance])
    return cmvn

def load_cmvn(cmvn_file):
    
    cmvn = _load_json_cmvn(cmvn_file)

    return cmvn[0], cmvn[1]