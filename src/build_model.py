
from typing import ContextManager
import numpy

import torch

import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules import loss

from data import EOS, SOS
from decoder import WFSTDecoder


inference_config = {
    "decoder_type":"wfst_decoder",
    "model_avg_num":10,
    "beam_size":10,
    "ctc_weight":0.5,
    "lm_weight":0.7,
    "lm_type": "",
    "wfst_graph":"corpus/aishell1/lm/graph/LG.fst",
    "acoustic_scale":6.0,
    "max_active":1000,
    "min_active":0,
    "wfst_beam": 30.0,
    "max_seq_len":100
  }

def mod_pad(x:torch.Tensor, mod:int):
    b, t, f = x.shape
    # print(b, t, f)
    pad = torch.zeros((b, (mod-t%mod)%mod, f), dtype=x.dtype, device=x.device)
    x = torch.cat((x, pad), dim=1)
    return x

def mod_ext(x:torch.Tensor, mod:int):
    x = x[:, ::mod, :]
    return x

def mod_cat(x:torch.Tensor, mod:int):
    b, t, f = x.shape
    x = torch.reshape(x, (b, t//mod, f*mod))
    return x



class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, eps=0.1):
        super(LabelSmoothingCrossEntropy, self).__init__()
        self.eps = eps
    
    def forward(self, output, target):
        c = output.size()[-1]
        assert c > 1
        log_preds = F.log_softmax(output, dim=-1)
        loss = -log_preds.sum(dim=-1)
        loss = loss.mean()
        return loss * self.eps/c + (1-self.eps) * F.nll_loss(log_preds, target, reduction='mean')

class LAS(nn.Module):
    def __init__(self, feat_dim:int, vocab:int, global_cmvn:torch.nn.Module = None, is_wfst=False):
        super(LAS, self).__init__()

        self.feat_dim = feat_dim
        self.vocab = vocab
        self.global_cmvn = global_cmvn
        self.encoder = Encoder(self.feat_dim*4) # *4
        self.decoder = Decoder(self.vocab, is_wfst=is_wfst)

        self.loss = LabelSmoothingCrossEntropy()
        self.sos = SOS
        self.eos = EOS

        self.is_wfst = is_wfst
        
    
    def forward(self, x:torch.Tensor, x_mask:torch.Tensor, y:torch.Tensor, y_mask:torch.Tensor):
        if self.global_cmvn is not None:
            x = self.global_cmvn(x)
        # print(x.shape)
        # assert False
        x = mod_pad(x, 4)
        x = mod_cat(x, 4)
        x_mask = mod_pad(x_mask, 4)
        x_mask = mod_ext(x_mask, 4)
        eout = self.encoder(x, x_mask)
        logit = self.decoder(eout, x_mask, y, y_mask)

        real = y_mask[:, 1:].gt(0)
        # real = y_mask[:,:].gt(0)
        label = y[:, 1:]  # [:, 1:]
        # print(label.shape)
        # print(logit.shape)
        # print(real)

        loss = self.loss(logit[real], label[real])
        return loss
    
    def predict(self, x:torch.Tensor, x_mask:torch.Tensor):
        assert 1 == x.shape[0]
        if self.global_cmvn is not None:
            x = self.global_cmvn(x)
        x = mod_pad(x, 4)
        x = mod_cat(x, 4)
        x_mask = mod_pad(x_mask, 4)
        x_mask = mod_ext(x_mask, 4)

        eout = self.encoder(x, x_mask)
        # print(eout.shape)
        if self.is_wfst:
            pred = self.decoder.wfst_decode(eout) # []
        else:
            pred = self.decoder.beamsearch(eout)

        return pred
    
    @torch.jit.export
    def sos_symbol(self) -> int:
        return self.sos 
    
    @torch.jit.export 
    def eos_symbol(self) -> int:
        return self.eos 





class Encoder(nn.Module):
    def __init__(self, feat_dim: int):
        super(Encoder, self).__init__()

        self.feat_dim = feat_dim
        self.proj_size = 256
        self.lstm_1 = nn.LSTM(input_size=feat_dim, hidden_size=512, num_layers=1, bias=True, batch_first=True)
        self.lstm_2 = nn.LSTM(input_size=512, hidden_size=512, num_layers=1, bias=True, batch_first=True)
        self.lstm_3 = nn.LSTM(input_size=512, hidden_size=512, num_layers=1, bias=True, batch_first=True)
        self.blstm = nn.LSTM(input_size=512, hidden_size=512, num_layers=1, bias=True, batch_first=True, bidirectional=True)

        self.conv_1 = nn.Conv2d(1024, 512, 1, 1, 0)
        self.relu_1 = nn.LeakyReLU(negative_slope=0.1)
        self.drop_1 = nn.Dropout(p=0.3)
        self.conv_2 = nn.Conv2d(512, self.proj_size, 1, 1, 0)

        nn.init.xavier_uniform_(self.conv_1.weight)
        nn.init.xavier_uniform_(self.conv_2.weight)
        nn.init.constant_(self.conv_1.bias, 0)
        nn.init.constant_(self.conv_2.bias, 0)

    def forward(self, x:torch.Tensor, x_mask: torch.Tensor):
        # x [B, T, D]


        x, _ = self.lstm_1(x)
        x, _ = self.lstm_2(x)
        x, _ = self.lstm_3(x)

        # print(x.shape)
        x = x * x_mask

        blstm_out, _ = self.blstm(x)
        blstm_out = blstm_out * x_mask

        blstm_out = blstm_out.permute(0, 2, 1) # [B, T, D]->[B, D, T]
        blstm_out = blstm_out.unsqueeze(dim=3) # [B, D, T]->[B, D, T, 1]

        conv = self.conv_1(blstm_out)
        conv = self.relu_1(conv)
        conv = self.drop_1(conv)
        conv = self.conv_2(conv)

        # [B, D, T, 1] -> [B, T, D]
        conv = conv.squeeze(dim=3)
        conv = conv.permute(0, 2, 1)

        eout = conv * x_mask

        return eout

class Decoder(nn.Module):
    def __init__(self, vocab: int, is_wfst=False):
        super(Decoder, self).__init__()

        self.vocab = vocab
        self.word_dim = 256
        self.hidden_size = 256
        self.proj_size = self.hidden_size
        self.embedding = nn.Embedding(vocab, self.word_dim)
        self.att_lstm = nn.LSTM(input_size=self.word_dim*2, hidden_size=self.hidden_size, num_layers=1, bias=True, batch_first=True)

        self.w = nn.Linear(self.word_dim, self.word_dim)
        self.v = nn.Linear(self.proj_size, self.word_dim)
        self.w_att_v = nn.Parameter(torch.randn([1, 1, self.word_dim]))

        nn.init.uniform_(self.w.weight, -0.05, 0.05)
        nn.init.uniform_(self.v.weight, -0.05, 0.05)
        nn.init.zeros_(self.w_att_v)
        
        self.dec_lstm = nn.LSTM(input_size=self.word_dim+self.proj_size, hidden_size=self.hidden_size, num_layers=1, bias=True, batch_first=True)
        self.dec_drop = nn.Dropout(p=0.3)
        self.classification = nn.Conv2d(self.word_dim+self.proj_size+self.proj_size, self.vocab, 1, 1, 0)

        nn.init.xavier_normal_(self.classification.weight)
        nn.init.zeros_(self.classification.bias.data)

        
        if is_wfst:
            self.decoder = WFSTDecoder(fst_path=inference_config['wfst_graph'],
                                        sos=SOS,
                                        eos=EOS,
                                        acoustic_scale=inference_config['acoustic_scale'],
                                        max_active=inference_config['max_active'],
                                        min_active=inference_config['min_active'],
                                        beam=inference_config['wfst_beam'],
                                        max_seq_len=inference_config['max_seq_len']
                                        )
            self.words = []
            import io 
            for line in io.open('corpus/aishell1/lm/graph/words.txt', 'r', encoding='utf-8').readlines():
                word = line.strip().split()[0]
                self.words.append(word)
            self.vocab_wfst = {}
            for line in open('corpus/aishell1/lm/dict/units.txt', 'r', encoding='utf-8').readlines():
                # print(line)
                char, idx = line.strip().split()[0], line.strip().split()[1]
                self.vocab_wfst[char] = int(idx)

    def forward(self, eout:torch.Tensor,
                    x_mask:torch.Tensor,
                    y:torch.Tensor,
                    y_mask:torch.Tensor):
        
        att_b = y.shape[0] # batch
        att_t = y.shape[1]-1 # the length of text  -1

        h_att_lstm = torch.zeros([1, att_b, self.proj_size], device=eout.device)
        c_att_lstm = torch.zeros([1, att_b, self.hidden_size], device=eout.device)
        context = torch.zeros([att_b, self.word_dim], device=eout.device)
        init_hc = (torch.zeros([1, att_b, self.proj_size], device=eout.device),
                    torch.zeros([1, att_b, self.hidden_size], device=eout.device))
        
        att_h = self.w(eout)
        embed = self.embedding(y[:, :-1]) # [b, vocab, word_dim] # 去除 eos
        # print(y.shape)
        # print(embed.shape)
        # assert False
        lstm_holder = []
        context_holder = []

        for i in range(att_t):
            # [b, word_dim] [b, word_dim] -> [b, word_dim * 2]
            body = torch.cat((embed[:,i,:], context), dim=1)
            body = torch.unsqueeze(body, dim=1) # [b, 1, 2*word_dim]
            y_att_lstm, (h_att_lstm, c_att_lstm) = self.att_lstm(body, (h_att_lstm, c_att_lstm))
            lstm_holder.append(y_att_lstm.squeeze(dim=1))

            state = self.v(y_att_lstm) # [b, 1, word_dim]
            # additive attention
            out = self.w_att_v * torch.tanh(state + att_h)
            scalar = out.sum(dim=2).unsqueeze(dim=2) # [b, 1]

            scalar = scalar + (x_mask-1)*1e30 # mask
            scalar = torch.softmax(scalar, dim=1)

            context = eout * scalar
            context = context.sum(dim=1)

            context_holder.append(context)

        lstms = torch.stack(lstm_holder, dim=1)
        contexts = torch.stack(context_holder, dim=1)
        lstms = lstms * y_mask[:,1:].unsqueeze(dim=2)
        contexts = contexts * y_mask[:, 1:].unsqueeze(dim=2)
        # lstms = lstms * y_mask[:,:].unsqueeze(dim=2)
        # contexts = contexts * y_mask[:, :].unsqueeze(dim=2)

        att_fea = torch.cat((lstms, contexts), dim=2)
        dout, _ = self.dec_lstm(att_fea, init_hc)
        dout = torch.cat((att_fea, dout), dim=2)
        dout = self.dec_drop(dout)
        dout = dout * y_mask[:,1:].unsqueeze(dim=2)
        # dout = dout * y_mask[:,:].unsqueeze(dim=2)
        dout = dout.permute(0,2,1).unsqueeze(dim=3)

        logit = self.classification(dout)

        logit = logit.squeeze(dim=3).permute(0,2,1)
        # print(logit.shape)
        # assert False

        return logit
    
    @torch.jit.export
    def decode_ont_step(self, y,
                        eout : torch.Tensor,
                        context : torch.Tensor,
                        h_att_lstm: torch.Tensor,
                        c_att_lstm: torch.Tensor,
                        init_h : torch.Tensor,
                        init_c : torch.Tensor,
                        att_h: torch.Tensor):
        """
        Args:
            y (torch.int64)
            eout (torch.Tensor) : 
            context (torch.Tensor):
            for att_lstm:
                h_att_lstm (torch.Tensor)
                c_att_lstm (torch.Tensor)
            for dec_lstm:
                init_h
                init_c
            att_h : w * h = self.w(eout)
        Returns:
            score : [b=1, 1, vocab_size]
            context, h_att_lstm, c_att_lstm, init_hc
        """
        
        # (context, h_att_lstm, c_att_lstm, init_hc) = _input

        # one step
        embed = self.embedding(y)
        # print(y.shape)
        # print(embed.shape)
        # assert False
        
        body = torch.cat((embed, context), dim=1)
        body = torch.unsqueeze(body, dim=1)
        


        y_att_lstm, (h_att_lstm, c_att_lstm) = self.att_lstm(body, (h_att_lstm, c_att_lstm))
        state = self.v(y_att_lstm)
        out = self.w_att_v * torch.tanh(state + att_h)
        scalar = out.sum(dim=2).unsqueeze(dim=2)
        scalar = torch.softmax(scalar, dim=1)

        context = eout * scalar
        context = context.sum(dim=1)

        att_fea = torch.cat((y_att_lstm, context.unsqueeze(dim=1)), dim=2)
        dout, init_hc = self.dec_lstm(att_fea, (init_h, init_c))
        dout = torch.cat((att_fea, dout), dim=2)
        dout = dout.permute(0,2,1).unsqueeze(dim=3)
        logit = self.classification(dout)
        logit = logit.squeeze(dim=3).permute(0,2,1)
        logit = logit*0.8
        score = torch.nn.functional.softmax(logit, dim=2)
        # print(score)
        # libtorch: cann't use numpy
        # score[score < numpy.exp(-1e30)] = numpy.exp(-1e30) 
        score = torch.log(score) #[1,1,4236]
        # print(score.shape)
        # print(context.shape)
        # print(h_att_lstm.shape)
        # print(c_att_lstm.shape)
        # print(init_h.shape)
        # print(init_c.shape)
        # assert False
        
        return score, context, h_att_lstm, c_att_lstm, init_hc

   
    def decode_one_step_wfst(self, eouts, cur_input, inner_packed_states):
        """
        for wfst decode
        Args:
            eout: ouputs of encoder ([1, xlen, xdim], [1, xlen, xdim])
            cur_input: input sequence , type: list [[id1, id2..], [id1, id2, ...]]
            inner_packed_states: inner states need to be record , type: tuple list [tuple, tuple, ..]
                            egs: (step, (context, h_att_lstm, c_att_lstm, init_h, init_c))
        Returns:
            score : log scores 
            inner_packed_states: inner states for next iterator
        """
        
        # (context, h_att_lstm, c_att_lstm, init_hc) = _input
        bs = len(cur_input) # beam size
        assert bs == len(inner_packed_states)
        # print('bs', bs)
        (eout, att_h) = eouts
        eout = eout.repeat(bs, 1, 1) 
        att_h = att_h.repeat(bs, 1, 1)
        (step, _ ) = inner_packed_states[0]
        step += 1
        # print(inner_packed_states)
        # assert False
        context, h_att_lstm, c_att_lstm, init_h, init_c = [], [], [], [], []
        for _iner in inner_packed_states:
            # print(_iner)
            # assert False
            (_, (_context, _h_att_lstm, _c_att_lstm, _init_h, _init_c)) = _iner
            # print('_contxt',_context.shape)
            context.append(_context)
            h_att_lstm.append(_h_att_lstm)
            c_att_lstm.append(_c_att_lstm)
            init_h.append(_init_h)
            init_c.append(_init_c)
        context = torch.cat(context, dim=0)
        # print('context',context.shape)
        # assert False
        h_att_lstm = torch.cat(h_att_lstm, dim=1)
        c_att_lstm = torch.cat(c_att_lstm, dim=1)
        init_h = torch.cat(init_h, dim=1)
        init_c = torch.cat(init_c, dim=1)

        # one step
        cur_input = torch.tensor(cur_input, device=eout.device)
        # print(cur_input)
        # print(cur_input[:,-1:])
        # print(cur_input[:, -1:].shape)
        embed = self.embedding(cur_input[:,-1])
        # print(embed.shape)
        
        body = torch.cat((embed, context), dim=1)
        # print(body.shape)
        body = torch.unsqueeze(body, dim=1)
        


        y_att_lstm, (h_att_lstm, c_att_lstm) = self.att_lstm(body, (h_att_lstm, c_att_lstm))
        state = self.v(y_att_lstm)
        out = self.w_att_v * torch.tanh(state + att_h)
        scalar = out.sum(dim=2).unsqueeze(dim=2)
        scalar = torch.softmax(scalar, dim=1)

        context = eout * scalar
        context = context.sum(dim=1)

        att_fea = torch.cat((y_att_lstm, context.unsqueeze(dim=1)), dim=2)
        dout, (init_h, init_c) = self.dec_lstm(att_fea, (init_h, init_c))
        dout = torch.cat((att_fea, dout), dim=2)
        dout = dout.permute(0,2,1).unsqueeze(dim=3)
        logit = self.classification(dout)
        logit = logit.squeeze(dim=3).permute(0,2,1)
        logit = logit*0.8
        score = torch.nn.functional.softmax(logit, dim=2)
        # print(score)
        # libtorch: cann't use numpy
        # score[score < numpy.exp(-1e30)] = numpy.exp(-1e30) 
        # print(context.shape)
        # print(h_att_lstm.shape)
        # print(c_att_lstm.shape)
        # print(init_h.shape)
        # print(init_c.shape)
        score = torch.log(score).squeeze(dim=1) # [bs, 1, vocabsize]
        
        # print(score.shape)
        # print(context[0, :].unsqueeze(dim=0).shape)
        # print(h_att_lstm[:,0,:].unsqueeze(dim=1).shape)
        # print(c_att_lstm[:,0,:].unsqueeze(dim=1).shape)
        
        # assert False

        # return score, context, h_att_lstm, c_att_lstm, init_hc
        return score.detach().cpu().numpy(), [(step, (context[b,:].unsqueeze(dim=0), 
                                                h_att_lstm[:,b,:].unsqueeze(dim=1), 
                                                c_att_lstm[:,b,:].unsqueeze(dim=1), 
                                                init_h[:,b,:].unsqueeze(dim=1), 
                                                init_c[:,b,:].unsqueeze(dim=1) )) for b in range(bs)]

    def beamsearch(self, eout:torch.Tensor):
        beam_size = 10
        max_utt_len = eout.shape[1]

        att_h = self.w(eout)
        # print(eout.shape)
        # print(att_h.shape)
        # assert False

        h_att_lstm = torch.zeros([1,1,self.proj_size], device=eout.device)
        c_att_lstm = torch.zeros([1,1,self.hidden_size], device=eout.device)
        context = torch.zeros([1, self.word_dim], device=eout.device)
        # init_hc = (torch.zeros([1,1,self.proj_size], device=eout.device),
        #             torch.zeros([1,1,self.hidden_size], device=eout.device))
        # for libtorch
        init_h = torch.zeros([1,1,self.proj_size], device=eout.device)
        init_c = torch.zeros([1,1,self.hidden_size], device=eout.device)

        beams = [{'h_att_lstm': h_att_lstm,
                'c_att_lstm': c_att_lstm,
                'init_hc': (init_h, init_c),
                'context': context,
                'preds': [torch.full([1], SOS, dtype=torch.int64, device=eout.device)], # the first predict word is <s>
                'scores': []}]
        
        for n in range(max_utt_len): # eout length
            beams_updates = []
            for i in range(len(beams)):
                b = beams[i]
                if b['preds'][-1].item() == EOS: # stop </s>
                    beams_updates.append(b)
                    continue
                else:
                    y = b['preds'][-1]
                    context = b['context']
                    h_att_lstm = b['h_att_lstm']
                    c_att_lstm = b['c_att_lstm']
                    (init_h, init_c) = b['init_hc']
                    
                    score, context, h_att_lstm, c_att_lstm, init_hc = self.decode_ont_step(y,
                                                                                        eout,
                                                                                        context=context,
                                                                                        h_att_lstm=h_att_lstm,
                                                                                        c_att_lstm=c_att_lstm,
                                                                                        init_h=init_h,
                                                                                        init_c=init_c,
                                                                                        att_h=att_h)

                    top_score, top_id = torch.topk(score, k=beam_size, dim=-1)
                    for j in range(beam_size):
                        b_branch = b.copy()
                        b_branch['context'] = context
                        b_branch['h_att_lstm'] = h_att_lstm
                        b_branch['c_att_lstm'] = c_att_lstm
                        b_branch['init_hc'] = init_hc
                        b_branch['preds'] = b_branch['preds'] + [top_id[0,0,j].reshape(1)]
                        b_branch['scores'] = b_branch['scores'] + [top_score[0,0,j].item()]
                        beams_updates.append(b_branch)
            beams = beams_updates
            # print(beams)
            # assert False
            beams = sorted(beams, key=lambda b : numpy.mean(b['scores']), reverse=True)
            beams = beams[:beam_size]
        b = beams[0]
        preds = b['preds'] # token_id
        preds = torch.stack(preds, dim=-1) # [1, seq_length]
        return preds

    def wfst_decode(self, eout:torch.Tensor):
        """
        wfst decode
        """
        batch_size = eout.size(0)
        assert batch_size == 1, 'batchsize must be one in test'
        att_h = self.w(eout)
        

        h_att_lstm = torch.zeros([1,1,self.proj_size], device=eout.device)
        c_att_lstm = torch.zeros([1,1,self.hidden_size], device=eout.device)
        context = torch.zeros([1, self.word_dim], device=eout.device)
        init_h = torch.zeros([1,1,self.proj_size], device=eout.device)
        init_c = torch.zeros([1,1,self.hidden_size], device=eout.device)

        initial_packed_states = (0, (context, h_att_lstm, c_att_lstm, init_h, init_c))
        # eout: [1, xlen, xdim]
        # att_h: [1, xlen, xdim]
        self.decoder.decode((eout, att_h), initial_packed_states, self.decode_one_step_wfst)

        words_pred_id = self.decoder.get_best_path()
        words_pred = ''.join([self.words[int(idx)] for idx in words_pred_id])
        preds = [self.vocab_wfst[pred] for pred in words_pred]
        
        # print(words_pred)
        # print(torch.tensor(preds))
        return torch.tensor(preds).unsqueeze(dim=0)
        