
# reference: athena-decoder-master
#
#

# NOTE: eps = 0 , <blank> = 0 for ctc in neural_sp
# so, 
# eps 0
# <blank> 1
# ...

"""WFST decoder for Seq2Seq model

Python version decoder which treat Seq2Seq models(attention, transformer, CTC)
as acoustic model. Record inner states of Seq2Seq models in token and execute
token passing algorithm in WFST decoding graph.
"""

import sys
import numpy as np
import openfst_python as fst

class LatticeArc:
    """Arc used in Token
    acoustic cost and graph cost are stored seperately.
    """
    def __init__(self, ilabel, olabel, weight, nextstate):
        self.ilabel = ilabel
        self.olabel = olabel
        self.weight = weight
        self.nextstate = nextstate

class Token:
    """Token used in token passing decode algorithm.
    The token can be linked by another token or None.
    The token record the linked arc, cost and inner packed states
    used in model
    """
    def __init__(self, arc, acoustic_cost, prev_tok=None, cur_label=None, inner_packed_states=None):
        self.prev_tok = prev_tok
        self.cur_label = cur_label
        self.inner_packed_states = inner_packed_states
        self.arc = LatticeArc(arc.ilabel, arc.olabel,
                (float(arc.weight), acoustic_cost), arc.nextstate)
        if prev_tok is not None:
            self.cost = prev_tok.cost + float(arc.weight) + acoustic_cost
        else:
            self.cost = float(arc.weight) + acoustic_cost
        self.rescaled_cost = -1.0 


class WFSTDecoder:
    """Decode the input samples using seq2seq models and WFST graph"""
    def __init__(self, fst_path, sos, eos, acoustic_scale=20.0,
            max_active=30, min_active=0, beam=30.0, max_seq_len=100):
        """Init decoder set some inner variance.

        Args:
            fst_path: path of WFST graph
        """
        self.cur_toks = {}
        self.prev_toks = {}
        self.completed_token_pool = []
        self.num_steps_decoded = -1
        self.beam_delta = 0.5
        self.fst = fst.Fst.read(fst_path)
        self.acoustic_scale = acoustic_scale
        self.max_active = max_active
        self.min_active = min_active
        self.beam = beam
        self.sos = sos
        self.eos = eos
        self.max_seq_len = max_seq_len

    def decode(self, encoder_outputs, initial_packed_states, inference_one_step_fn):
        """using seq2seq model and WFST graph to decode input utterance

        Args:
            encoder_outputs: outputs of encoder for encoder-decoder structure model.
                             or just CTC outputs for ctc model.
            inference_one_step_fn: callback function of seq2seq model. the function
                                   take encoder_outputs,input label and inner states,
                                   then produce logscores and inner states.
                                   the function's signature is: logscores, packed_inner_states =
                                   fn(encoder_outputs, label_input, packed_inner_states)
            initial_packed_states: initial packed states for inference_one_step_fn callback function
        """
        self.init_decoding(initial_packed_states)
        while not self.end_detect():
            self.prev_toks = self.cur_toks
            self.cur_toks = {}
            weight_cutoff = self.process_emitting(encoder_outputs, inference_one_step_fn)
            self.process_nonemitting(weight_cutoff)

    def end_detect(self):
        """determine whether to stop propagating"""
        if self.cur_toks and self.num_steps_decoded < self.max_seq_len:
            return False
        else:
            return True

    def init_decoding(self, initial_packed_states):
        """Init decoding states for every input utterance

        Args:
            initial_packed_states: initial packed states for callback function
        """
        self.cur_toks = {}
        self.prev_toks = {}
        self.completed_token_pool = []
        start_state = self.fst.start()
        # print('start_state', start_state)
        assert start_state != -1
        # assert start_state != 0
        dummy_arc = LatticeArc(0, 0, 0.0, start_state)
        self.cur_toks[start_state] = Token(dummy_arc, 0.0, None, [self.sos], initial_packed_states)
        # print(self.cur_toks)
        self.num_steps_decoded = 0
        self.process_nonemitting(float('inf'))

    def deal_completed_token(self, state, eos_score):
        """deal completed token and rescale scores

        Args:
            state: state of the completed token
            eos_score: acoustic score of eos
        """
        tok = self.prev_toks[state]
        if self.fst.final(state).to_string() != fst.Weight.Zero('tropical').to_string():
            tok.rescaled_cost = ((tok.cost + (-eos_score) +
                float(self.fst.final(state)))/self.num_steps_decoded)
            self.completed_token_pool.append(tok)
        queue = []
        tmp_toks = {}
        queue.append(state)
        tmp_toks[state] = tok
        while queue:
            state = queue.pop()
            tok = tmp_toks[state]
            for arc in self.fst.arcs(state):
                if arc.ilabel == 0: # 0->-1 eps
                    new_tok = Token(arc, 0.0, tok, tok.cur_label, tok.inner_packed_states)
                    if self.fst.final(arc.nextstate).to_string() != fst.Weight.Zero('tropical').to_string():
                        new_tok.rescaled_cost = ((new_tok.cost +
                            float(self.fst.final(arc.nextstate)))/self.num_steps_decoded)
                        self.completed_token_pool.append(new_tok)
                    else:
                        tmp_toks[arc.nextstate] = new_tok
                        queue.append(arc.nextstate)

    def process_emitting(self, encoder_outputs, inference_one_step_fn):
        """Process one step emitting states using callback function

        Args:
            encoder_outputs: encoder outputs
            inference_one_step_fn: callback function

        Returns:
            next_weight_cutoff: cutoff for next step
        """
        state2id = {}
        cand_seqs = []
        inner_packed_states_array = []
        # print('prev_toks',self.prev_toks)
        for idx, state in enumerate(self.prev_toks):
            # print(idx, state)
            state2id[state] = idx
            cand_seqs.append(self.prev_toks[state].cur_label)
            inner_packed_states_array.append(self.prev_toks[state].inner_packed_states)
        # 自回归推理
        # print(cand_seqs)
        # print('cand_seqs:',cand_seqs.shape)
        # assert False
        all_log_scores, inner_packed_states_array = inference_one_step_fn(encoder_outputs,
                cand_seqs, inner_packed_states_array)
        weight_cutoff, adaptive_beam, best_token = self.get_cutoff()
        # print(weight_cutoff,adaptive_beam)
        next_weight_cutoff = float('inf')
        if best_token is not None:
            for arc in self.fst.arcs(best_token.arc.nextstate):
                if arc.ilabel != 0: # 0-> -1 eps # 只处理输入标签不为<eps>的情况
                    # arc.ilabel - 1
                    ac_cost = (-all_log_scores[state2id[best_token.arc.nextstate]][arc.ilabel-1] *
                            self.acoustic_scale)
                    new_weight = float(arc.weight) + best_token.cost + ac_cost
                    # 更新界线值，使剪枝阈值在一个合理的范围内
                    if new_weight + adaptive_beam < next_weight_cutoff:
                        next_weight_cutoff = new_weight + adaptive_beam
        for state, tok in self.prev_toks.items():
            seq_id = state2id[state]
            # print(seq_id)
            if tok.cost <= weight_cutoff:
                # 识别结束
                if self.eos == np.argmax(all_log_scores[seq_id]):
                    self.deal_completed_token(state, all_log_scores[seq_id][self.eos])
                    continue
                for arc in self.fst.arcs(state):
                    if arc.ilabel != 0: # 0->-1 eps # 只处理输入标签不为<eps>的情况
                        # TODO check ac_cost
                        # arc.ilabel - 1
                        # print(all_log_scores.shape)
                        # print(all_log_scores[seq_id][arc.ilabel-1])
                        ac_cost = -all_log_scores[seq_id][arc.ilabel-1] * self.acoustic_scale
                        # print(ac_cost)
                        # print(float(arc.weight))
                        # print(tok.cost)
                        # print('--------')
                        # assert False
                        new_weight = float(arc.weight) + tok.cost + ac_cost
                        if new_weight < next_weight_cutoff:
                            # 基于当前token,建立新的token
                            # arc.ilabel - 1
                            new_tok = Token(arc, ac_cost, tok, tok.cur_label+[arc.ilabel-1],
                                    inner_packed_states_array[seq_id])
                            # 更新界线值，使剪枝阈值在一个合理的范围内
                            if new_weight + adaptive_beam < next_weight_cutoff:
                                next_weight_cutoff = new_weight + adaptive_beam
                            
                            if arc.nextstate in self.cur_toks:
                                # 如果下一个状态的token不如新token代价小，则用新token替换
                                if self.cur_toks[arc.nextstate].cost > new_tok.cost:
                                    self.cur_toks[arc.nextstate] = new_tok
                            else:
                                # 如果下一个状态还没有token，则将新token放在下一个状态上
                                self.cur_toks[arc.nextstate] = new_tok
        self.num_steps_decoded += 1
        return next_weight_cutoff

    def process_nonemitting(self, cutoff):
        """Process one step nonemitting states

        """
        queue = []
        for state, _ in self.cur_toks.items():
            queue.append(state)
        while queue:
            state = queue.pop()
            tok = self.cur_toks[state]
            if tok.cost > cutoff: # 代价过大 剪枝
                continue
            for arc in self.fst.arcs(state):
                if arc.ilabel == 0: # 0 -> -1  # eps状态
                    new_tok = Token(arc, 0.0, tok, tok.cur_label, tok.inner_packed_states)
                    if new_tok.cost < cutoff:
                        if arc.nextstate in self.cur_toks:
                            if self.cur_toks[arc.nextstate].cost > new_tok.cost:
                                self.cur_toks[arc.nextstate] = new_tok
                                queue.append(arc.nextstate)
                        else:
                            self.cur_toks[arc.nextstate] = new_tok
                            queue.append(arc.nextstate)

    def get_cutoff(self):
        """get cutoff used in current and next step

        Returns:
            beam_cutoff: beam cutoff
            adaptive_beam: adaptive beam
            best_token: best token this step
        """
        best_cost = float('inf')
        best_token = None
        if(self.max_active == sys.maxsize
                and self.min_active == 0):
            for _, tok in self.prev_toks.items():
                if tok.cost < best_cost:
                    best_cost = tok.cost # 代价最小
                    best_token = tok
                adaptive_beam = self.beam
                beam_cutoff = best_cost + self.beam
                return beam_cutoff, adaptive_beam, best_token
        else:
            tmp_array = []
            for _, tok in self.prev_toks.items():
                tmp_array.append(tok.cost)
                if tok.cost < best_cost:
                    best_cost = tok.cost
                    best_token = tok
            beam_cutoff = best_cost + self.beam
            min_active_cutoff = float('inf')
            max_active_cutoff = float('inf')
            if len(tmp_array) > self.max_active: # 超过最大候选列表
                np_tmp_array = np.array(tmp_array)
                k = self.max_active
                max_active_cutoff = np_tmp_array[np.argpartition(np_tmp_array, k-1)[k-1]] # 第k小 最大
            # 
                # print(max_active_cutoff)
                # print(np_tmp_array)
                # assert False, 'ccc'
            if max_active_cutoff < beam_cutoff:
                adaptive_beam = max_active_cutoff - best_cost + self.beam_delta
                return max_active_cutoff, adaptive_beam, best_token
            # TODO min_active max_active
            if len(tmp_array) > self.min_active:
                np_tmp_array = np.array(tmp_array)
                k = self.min_active
                if k == 0:
                    min_active_cutoff = best_cost
                else:
                    min_active_cutoff = np_tmp_array[np.argpartition(np_tmp_array, k-1)[k-1]] # 第k小
            if min_active_cutoff > beam_cutoff:
                adaptive_beam = min_active_cutoff - best_cost + self.beam_delta
                return min_active_cutoff, adaptive_beam, best_token
            else:
                adaptive_beam = self.beam
                return beam_cutoff, adaptive_beam, best_token

    def get_best_path(self):
        """get decoding result in best completed path

        Returns:
            ans: id array of decoding results
        """
        ans = []
        if not self.completed_token_pool:
            return ans
        best_completed_tok = self.completed_token_pool[0]
        for tok in self.completed_token_pool:
            if best_completed_tok.rescaled_cost > tok.rescaled_cost:
                best_completed_tok = tok
        arcs_reverse = []
        tok = best_completed_tok
        while tok:
            arcs_reverse.append(tok.arc)
            tok = tok.prev_tok
        assert arcs_reverse[-1].nextstate == self.fst.start()
        arcs_reverse.pop()
        for arc in arcs_reverse[::-1]:
            if arc.olabel != 0: # 0 -> -1 eps
                ans.append(arc.olabel)
        return ans
