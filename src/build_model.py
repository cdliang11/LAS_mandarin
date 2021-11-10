from math import e
import numpy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules import loss


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
    def __init__(self, feat_dim:int, vocab:int, global_cmvn:torch.nn.Module = None):
        super(LAS, self).__init__()

        self.feat_dim = feat_dim
        self.vocab = vocab
        self.global_cmvn = global_cmvn
        self.encoder = Encoder(self.feat_dim*4) # *4
        self.decoder = Decoder(self.vocab)

        self.loss = LabelSmoothingCrossEntropy()
    
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
        pred = self.decoder.beamsearch(eout)

        return pred

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
    def __init__(self, vocab: int):
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
        # print(embed.shape)

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

        return logit
    
    def beamsearch(self, eout:torch.Tensor):
        beam_size = 10
        max_utt_len = eout.shape[1]

        att_h = self.w(eout)

        h_att_lstm = torch.zeros([1,1,self.proj_size], device=eout.device)
        c_att_lstm = torch.zeros([1,1,self.hidden_size], device=eout.device)
        context = torch.zeros([1, self.word_dim], device=eout.device)
        init_hc = (torch.zeros([1,1,self.proj_size], device=eout.device),
                    torch.zeros([1,1,self.hidden_size], device=eout.device))

        beams = [{'h_att_lstm': h_att_lstm,
                'c_att_lstm': c_att_lstm,
                'init_hc': init_hc,
                'context': context,
                'preds': [torch.full([1], 4233, dtype=torch.int64, device=eout.device)], # the first predict word is <s>
                'scores': []}]
        
        for n in range(max_utt_len): # eout length
            beams_updates = []
            for i in range(len(beams)):
                b = beams[i]
                if b['preds'][-1].item() == 4234: # stop </s>
                    beams_updates.append(b)
                    continue
                else:
                    y = b['preds'][-1]
                    embed = self.embedding(y)

                    context = b['context']
                    h_att_lstm = b['h_att_lstm']
                    c_att_lstm = b['c_att_lstm']
                    init_hc = b['init_hc']

                    body = torch.cat((embed, context), dim=1)
                    body = torch.unsqueeze(body, dim=1)
                    # print(embed.shape)
                    # print(context.shape)
                    # print(body.shape)
                    y_att_lstm, (h_att_lstm, c_att_lstm) = self.att_lstm(body, (h_att_lstm, c_att_lstm))

                    state = self.v(y_att_lstm)

                    out = self.w_att_v * torch.tanh(state+att_h)
                    scalar = out.sum(dim=2).unsqueeze(dim=2)

                    scalar = torch.softmax(scalar, dim=1)

                    context = eout * scalar
                    context = context.sum(dim=1)

                    att_fea = torch.cat((y_att_lstm, context.unsqueeze(dim=1)), dim=2)
                    dout, init_hc = self.dec_lstm(att_fea, init_hc)
                    dout = torch.cat((att_fea, dout), dim=2)
                    dout = dout.permute(0,2,1).unsqueeze(dim=3)
                    # print(dout.shape)
                    logit = self.classification(dout)
                    logit = logit.squeeze(dim=3).permute(0,2,1)
                    logit = logit * 0.8
                    score = torch.nn.functional.softmax(logit, dim=2)
                    score[score < numpy.exp(-1e30)] = numpy.exp(-1e30)
                    score = torch.log(score)
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
            beams = sorted(beams, key=lambda b : numpy.mean(b['scores']), reverse=True)
            beams = beams[:beam_size]
        b = beams[0]
        preds = b['preds'] # token_id
        preds = torch.stack(preds, dim=-1) # [1, seq_length]
        return preds