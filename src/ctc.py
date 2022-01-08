
import torch 
import torch.nn.functional as F
from typeguard import check_argument_types
import numpy as np

from utils import tensor2np

LOG_0 = -1e10 
LOG_1 = 0


class CTC(torch.nn.Module):
    """CTC model"""
    def __init__(self, odim: int, enc_projs: int, 
                dropout_rate: float = 0.0, 
                reduce: bool = True):
        """CTC model 
        Args:
            odim:
        """
        assert check_argument_types()
        super().__init__()
        self.enc_projs = enc_projs
        self.dropout_rate = dropout_rate 
        self.ctc_lo = torch.nn.Linear(self.enc_projs, odim)

        reduction_type = 'sum' if reduce else 'none'
        self.ctc_loss = torch.nn.CTCLoss(reduction=reduction_type)


    def forward(self, hs_pad: torch.Tensor,
                hlens: torch.Tensor,
                ys_pad: torch.Tensor, 
                ys_lens: torch.Tensor) -> torch.Tensor:
        """ Calulate CTC loss
        Args:
            hs_pad: batch of padded hidden state sequences (B, T, D)
            hlens: batch of lengths of hidden state sequences (B)
            ys_pad: batch of padded character id sequence tensor (B, Lmax)
            ys_lens: batch of lengths of character sequence (B)
            
        """
        # [B, L, nproj] -> [B, L, nvocab]
        ys_hat = self.ctc_lo(F.dropout(hs_pad, p=self.dropout_rate))
        # [B, L, nvocab] -> [L, B, nvocab]
        ys_hat = ys_hat.transpose(0, 1)
        ys_hat = ys_hat.log_softmax(2)
        loss = self.ctc_loss(ys_hat, ys_pad, hlens, ys_lens)
        # batch-size average
        loss = loss / ys_hat.size(1)
        return loss 
    
    def log_softmax(self, hs_pad: torch.Tensor) -> torch.Tensor:
        """log_softmax of frame activations 
        """
        return F.log_softmax(self.ctc_lo(hs_pad), dim=2)
    
    def argmax(self, hs_pad: torch.Tensor) -> torch.Tensor:
        """argmax of frame activations
        """
        return torch.argmax(self.ctc_lo(hs_pad), dim=2)



class CTCPrefixScore(object):
    """Compute CTC label sequence scores
    which is based on Algorithm 2 in WATANABE et al.
    "HYBRID CTC/ATTENTION ARCHITECTURE FOR END-TO-END SPEECH RECOGNITION,"
    but extended to efficiently compute the probabilities of multiple labels
    simultaneously

    [Reference]:
        https://github.com/espnet/espnet
    """
    def __init__(self, log_probs, blank, eos, truncate=False):
        """
        Args:
            log_probs: 
            blank (int): index of <blank>
            eos (int): index of <eos>
            truncate (bool): restart prefix search from the previous CTC spike
        """
        self.blank = blank 
        self.eos = eos 
        self.xlen_prev = 0 
        self.xlen = len(log_probs)
        self.log_probs = log_probs
        self.log0 = LOG_0

        self.truncate = truncate
        self.offset = 0 
    
    def initial_state(self):
        """
        obtation an initial CTC state 
        Returns:
            ctc_state (np.ndarray): [T, 2]
        """
        r = np.full((self.xlen, 2), self.log0, dtype=np.float32)
        r[0,1] = self.log_probs[0, self.blank]
        for i in range(1, self.xlen):
            r[i, 1] = r[i-1, 1] + self.log_probs[i, self.blank]
        return r 
    
    def __call__(self, hyp, cs, r_prev):
        """
        Compute CTC prefix scores for next labels 
        Args:
            hyp: prefix label sequences
            cs (np.ndarray) : array of next labels. A tensor of size [beam_width]
            r_prev (np.ndarray): previous CTC state [T, 2]
        Returns:
            ctc_scores (np.ndarray): [beam_width]
            ctc_states (np.ndarray): [beam_width, T, 2]
        """
        beam_width = len(cs)
        # initialize ctc states 
        ylen = len(hyp) - 1 # ingore sos
        # new ctc states are prepared as a frame x (n or b) 
        r = np.ndarray((self.xlen, 2, beam_width), dtype=np.float32)
        # print(self.log_probs.shape)
        xs = self.log_probs[:, cs]
        if ylen == 0:
            r[0,0] = xs[0]
            r[0,1] = self.log0 
        else:
            r[ylen-1] = self.log0
        
        # prepare forward probabilities for the last label 
        r_sum = np.logaddexp(r_prev[:, 0], r_prev[:, 1]) # log
        last = hyp[-1] 
        if ylen > 0 and last in cs:
            log_phi = np.ndarray((self.xlen, beam_width), dtype=np.float32) 
            for k in range(beam_width):
                log_phi[:, k] = r_sum if cs[k] != last else r_prev[:, 1]
        else:
            log_phi = r_sum # [T]
        
        # compute forward probabilities 
        # and log prefix probabilities log(psi)
        start = max(ylen, 1)
        log_psi = r[start - 1, 0]
        for t in range(start, self.xlen):
            # non-blank 
            # print(xs)
            # print(r[t-1, 0])
            # print(log_phi[t-1])
            r[t, 0] = np.logaddexp(r[t-1, 0], log_phi[t-1]) + xs[t]
            # blank
            r[t, 1] = np.logaddexp(r[t-1, 0], r[t-1, 1]) + self.log_probs[t, self.blank]
            log_psi = np.logaddexp(log_psi, log_phi[t-1] + xs[t])
        
        # get p(...eos|x) that ends with the prefix itself
        eos_pos = np.where(cs == self.eos)[0] 
        if len(eos_pos) > 0:
            log_psi[eos_pos] = r_sum[-1] # 
        
        # return the log prefix probability and ctc states, where the label axis
        # of the CTC states is moved to the first axis to slice it easily 
        return log_psi, np.rollaxis(r, 2)
        
        

def add_ctc_score(hyp, topk_ids, ctc_state, total_scores_topk,
                ctc_prefix_scorer, beam_size, ctc_weight):
    """
    hyp: label sequence
    topk_ids: [1,1,10] torch.Tensor
    ctc_state: [T,2]
    """
    # print(hyp)
    # print(ctc_state.shape)
    # print(beam_size)


    if ctc_prefix_scorer is None:
        return None, topk_ids.new_zeros(beam_size), total_scores_topk
    
    ctc_scores, new_ctc_states = ctc_prefix_scorer(hyp, tensor2np(topk_ids[0][0]), ctc_state)
    total_scores_ctc = torch.from_numpy(ctc_scores).to(topk_ids.device) # fix multi-gpu
    # total_scores_ctc = total_scores_ctc.unsqueeze(0).unsqueeze(0)
    # print(total_scores_ctc)
    # print(total_scores_topk)
    

    total_scores_topk += total_scores_ctc * ctc_weight  # boosting [1,1,beam_size]
    # sort again 
    # print(total_scores_topk)
    total_scores_topk, joint_ids_topk = torch.topk(
        total_scores_topk, k=beam_size, dim=-1, largest=True, sorted=True)
    # print(total_scores_topk)
    # print(joint_ids_topk)
    # assert False
    # print(topk_ids)
    topk_ids = topk_ids[:, :, joint_ids_topk[0][0]] # [1,1,beam_size]
    # print(topk_ids)
    # assert False 
    # print(new_ctc_states)
    new_ctc_states = new_ctc_states[tensor2np(joint_ids_topk[0][0])]
    # print(new_ctc_states)
    # assert False
    return new_ctc_states, ctc_scores, total_scores_topk, topk_ids