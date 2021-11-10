
# for test

import copy
from json import load
import os
from numpy import tri 
import torch
from data import AudioDataset, collatefunc
from torch.utils.data import DataLoader
import build_model
import editdistance
import time
import argparse
import yaml
from utils import load_cmvn, GlobalCMVN

from train import writelog
from utils import make_pad_mask

def decode(vocab:dict, token_id:torch.Tensor):
    bs = token_id.shape[0] # batch_size
    ylen = token_id.shape[1]

    text = [] # [bs, 1]
    for i in range(bs):
        utt = []
        for j in range(ylen):
            idx = token_id[i][j].item()
            if idx == 4234:
                break
            if 3 <= idx <= 4232:
                utt.append(vocab[idx])
        utt = ''.join(utt)
        text.append(utt)
    return text

def main(args, conf):
    vocab = {}
    for line in open(args.vocab, 'r', encoding='utf-8'):
        word = line.strip().split()
        vocab[int(word[1])] = word[0]
    vocab_size = len(vocab)
    print('vocab_size: ', vocab_size)

    if args.cmvn_file is not None:
        mean, istd = load_cmvn(args.cmvn_file)
        global_cmvn = GlobalCMVN(torch.from_numpy(mean).float(),
                                torch.from_numpy(istd).float())
        print('global cmvn')
    else:
        global_cmvn = None
        print('no global cmvn')

    save_dir = args.save_dir # model path
    test_iter = args.test_iter
    model = build_model.LAS(feat_dim=80, vocab=vocab_size, global_cmvn=global_cmvn)
    info = torch.load(os.path.join(save_dir, "iter%d.pth"%test_iter), map_location=torch.device('cpu'))
    print(os.path.join(save_dir, 'iter%d.pth'%test_iter))
    model.load_state_dict(info['weights'])
    model.cuda()
    model.eval()

    num_params = 0
    for param in model.parameters():
        num_params += param.numel()
    print('param num: %d' % num_params)

    # test dataload
    test_collate_conf = copy.deepcopy(conf['data']['collate_fn'])
    test_collate_conf['spec_aug'] = False
    test_collate_conf['speed_perturb'] = False
    dataset_conf = conf['data']['dataset']
    dataset_conf['batch_size'] = 1
    dataset_conf['dynamic_batch'] = False

    test_dataset = AudioDataset(args.test_data_file, **dataset_conf, is_test=True)
    test_dataloader = DataLoader(test_dataset,
                                batch_size=1,
                                shuffle=False,
                                sampler=None,
                                num_workers=0,
                                collate_fn=collatefunc(**test_collate_conf))

    total_tokens = 0
    total_error = 0

    with torch.no_grad():
        st_time = time.time()
        cur_log = os.path.join(save_dir, 'test.log')
        if os.path.isfile(cur_log):
            os.remove(cur_log)
        writelog(cur_log, "iter%d.pth"%test_iter)
        writelog(cur_log, '--------------------------------------------------------')
        cnt_uttr = 0
        for i, batch in enumerate(test_dataloader):
            uttrs, feats, labels, xlens, ylens = batch
            xlens = xlens.cuda()
            feats = feats.cuda()
            labels = labels
            
            # print(xlens)
            # print(feats.shape)
            x_mask = make_pad_mask(xlens).unsqueeze(dim=-1).detach()
            # print(x_mask)
            labels = labels.long()

            preds_batch = model.predict(feats, x_mask)
            # print(preds_batch)
            # print(labels)
            # assert False
            uttr_pred = decode(vocab, preds_batch)
            uttr_label = decode(vocab, labels)
            pred = list(uttr_pred[0])
            label = list(uttr_label[0])
            total_error += editdistance.eval(label, pred)
            total_tokens += len(label)
            cnt_uttr += 1
            writelog(cur_log, 'utt[%d], current wer: %f' % (cnt_uttr, total_error / (total_tokens+1e-20)))
            writelog(cur_log, 'label: %s' % uttr_label)
            writelog(cur_log, 'pred: %s'%uttr_pred)
            # writelog(cur_log, '\n')
    
    writelog(cur_log, '----------------------------------------------------')
    writelog(cur_log, 'test iters: %d, total word: %d, error word: %d, wer: %f' %
            (test_iter, total_tokens, total_error, total_error/(total_tokens+1e-20)))
    



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='las test')
    parser.add_argument('--save_dir', type=str, default='')
    parser.add_argument('--conf_path', type=str, default='')
    parser.add_argument('--test_data_file', type=str, default='')
    parser.add_argument('--vocab', type=str, default='')
    parser.add_argument('--test_iter', type=int, default=0)
    parser.add_argument('--cmvn_file', type=str, default='')
    args = parser.parse_args()

    with open(args.conf_path, 'r', encoding='utf-8') as f:
        conf = yaml.load(f, Loader=yaml.FullLoader)
    main(args, conf)
