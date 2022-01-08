
import os
import time 
import numpy as np
from numpy.core.records import array
import torch
from torch import optim
from torch.autograd import grad
import yaml
import build_model
from data import collatefunc, AudioDataset
from torch.utils.data import DataLoader
from data import IGNORE_ID
from utils import make_pad_mask, load_cmvn, GlobalCMVN
import copy

import argparse

def writelog(path:str, log:str):
    f = open(path, 'a')
    f.write(log)
    f.write('\n')
    f.close()

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def set_lr(optimizer, new_lr: float):
    for param_group in optimizer.param_groups:
        param_group['lr'] = new_lr

def log_history(save_path, message):
    fname = os.path.join(save_path, 'history.csv')
    if not os.path.exists(fname):
        with open(fname, 'w') as f:
            f.write('datetime,iter,learning rate,train loss,dev loss\n')
            f.write("%s\n"%message)
    else:
        with open(fname, 'a') as f:
            f.write("%s\n" % message)

def save_checkpoint(filename, save_path, cur_iter, model, optimizer, scheduler):
    filename = os.path.join(save_path, filename)
    info = {'cur_iter':cur_iter,
            'weights':model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict()}
    
    torch.save(info, filename)

def main(args, conf):
    vocab = {}
    for line in open(args.vocab, 'r', encoding='utf-8'):
        word = line.strip().split()
        vocab[int(word[1])] = word[0]
    vocab_size = len(vocab)
    print('vocab_size: ', vocab_size)
    # print(vocab)

    if args.cmvn_file is not None:
        mean, istd = load_cmvn(args.cmvn_file)
        global_cmvn = GlobalCMVN(torch.from_numpy(mean).float(),
                                torch.from_numpy(istd).float())
        print('global cmvn')
    else:
        global_cmvn = None
        print('no global cmvn')


    # define train 
    # TODO: use yaml 
    tol_iter = conf['train']['tol_iter']
    init_lr = 3e-4
    base_lr = conf['train']['base_lr']
    grad_clip = 5.0
    save_dir = args.save_dir

    model = build_model.LAS(feat_dim=conf['data']['collate_fn']['feature_conf']['mel_bins'], 
                            vocab=vocab_size, 
                            global_cmvn=global_cmvn,
                            ctc_weight=0.3)
    optimizer = torch.optim.Adam(model.parameters(), lr=base_lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer,
                                                        mode='min',
                                                        factor=0.5,
                                                        patience=5,
                                                        threshold=0.01,
                                                        min_lr=1e-6)
    print(save_dir)
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    
    print(model)
    num_params = sum(p.numel() for p in model.parameters())
    print("The number of model params: {}".format(num_params))
    
    

    # init weight
    if args.init_model:
        print('load init_model from %s' % args.init_model)
        info = torch.load(args.init_model)
        model.load_state_dict(info['weights'])
    
    # resume model
    cur_iter = 0
    while os.path.exists(os.path.join(save_dir, 'iter%d.pth'%cur_iter)):
        if not os.path.exists(os.path.join(save_dir, 'iter%d.pth'%(cur_iter+1))):
            info = torch.load(os.path.join(save_dir, 'iter%d.pth'%cur_iter))
            cur_iter = info['cur_iter']
            model.load_state_dict(info['weights'])
            optimizer.load_state_dict(info['optimizer'])
            for state in optimizer.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.cuda()
            
            scheduler.load_state_dict(info['scheduler'])
        cur_iter += 1
    
    model.cuda()

    # dataload

    dataset_conf = conf['data']['dataset']
    train_dataset = AudioDataset(args.train_data_file, **dataset_conf, is_test=False)
    train_dataloader = DataLoader(train_dataset,
                                batch_size=1,
                                shuffle=True,
                                sampler=None,
                                num_workers=args.n_works,
                                collate_fn=collatefunc(**conf['data']['collate_fn']))

    dev_collate_conf = copy.deepcopy(conf['data']['collate_fn'])
    dev_collate_conf['spec_aug'] = False
    dev_collate_conf['speed_perturb'] = False

    dev_dataset = AudioDataset(args.dev_data_file, **dataset_conf, is_test=False)
    dev_dataloader = DataLoader(dev_dataset,
                                batch_size=1,
                                shuffle=True,
                                sampler=None,
                                num_workers=args.n_works,
                                collate_fn=collatefunc(**dev_collate_conf))
    

    while cur_iter < tol_iter:
        # epoch 
        st_time = time.time()
        cur_log = os.path.join(save_dir, 'iter%d.log'%cur_iter)
        if os.path.isfile(cur_log):
            os.remove(cur_log)
        
        cur_lr = get_lr(optimizer)
        iter_log = 'cur_iter=%d, cur_lr=%f' % (cur_iter, cur_lr)
        writelog(cur_log, '--------------------------------------------------------------')
        writelog(cur_log, iter_log)
        writelog(cur_log, '--------------------------------------------------------------')

        # train
        model.train()
        train_loss = []
        total_uttr_train = 0
        for i, batch in enumerate(train_dataloader):
            # batch : uttrs, feats_pad, labels_pad, xlens, ylens, _y, _ylens
            # _y and _ylens: dont have <s> and </s>
            uttrs, feats, labels, xlens, ylens, _y, _ylens = batch
            # print(labels.shape)
            # assert False
            feats = feats.cuda().detach()
            labels = labels.cuda().detach()
            xlens = xlens.cuda().detach()
            ylens = ylens.cuda().detach()
            _y = _y.cuda().detach()
            _ylens = _ylens.cuda().detach()
            num_uttr = ylens.size(0)
            if num_uttr == 0:
                continue
                
            # print(labels)
            # print(_y)
            # print(ylens)
            # print(_ylens)
            # assert False
            
            max_xlen = torch.max(xlens)
            x_mask = make_pad_mask(xlens).unsqueeze(dim=-1).detach()
            # print(x_mask.shape)
            # assert False
            
            # TODO:  make_pad_mask
            max_ylen = torch.max(ylens)
            labels[torch.where(labels<0)] = 2 # pad id
            labels = labels.long().detach()
            y_mask = torch.zeros_like(labels).cuda()
            y_mask[torch.where(labels != 2)] = 1
            _y[torch.where(_y < 0)] = 2 # pad id
            _y = _y.long().detach()
            # print(labels)
            # print(y_mask)

            # assert False
            loss, loss_att, loss_ctc = model(feats, x_mask, labels, y_mask, _y, _ylens)
            loss_att = loss_att if loss_att is not None else 0
            loss_ctc = loss_ctc if loss_ctc is not None else 0

            train_loss.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

            ed_time = time.time()
            total_uttr_train += num_uttr
            # read_utt, total_utt = train_data.Progress()
            train_log = "propossed=[%d], meanloss=%f (cur: loss: %f, loss_att: %f, loss_ctc: %f), duration_batch=%f" % \
                        (total_uttr_train, np.mean(train_loss), loss.item(), loss_att.item(), loss_ctc.item(), ed_time - st_time)
            writelog(cur_log, train_log)
        writelog(cur_log, '---------------------------------------------------------')

        # dev
        model.eval()
        dev_loss = []
        total_uttr_dev = 0
        for i, batch in enumerate(dev_dataloader):
            st_time = time.time()
            uttrs, feats, labels, xlens, ylens, _y, _ylens = batch
            feats = feats.cuda()
            labels = labels.cuda()
            xlens = xlens.cuda()
            ylens = ylens.cuda()
            _y = _y.cuda().detach()
            _ylens = _ylens.cuda().detach()

            num_uttr = ylens.size(0)
            if num_uttr == 0:
                continue
            
            max_xlen = torch.max(xlens)
            x_mask = make_pad_mask(xlens).unsqueeze(dim=-1).detach()
            # print(x_mask)
            # assert False
            
            # TODO:  make_pad_mask
            max_ylen = torch.max(ylens)
            labels[torch.where(labels<0)] = 2 # pad id
            labels = labels.long()
            y_mask = torch.zeros_like(labels).cuda()
            y_mask[torch.where(labels != 2)] = 1
            _y[torch.where(_y < 0)] = 2 # pad id 
            _y = _y.long().detach()

            loss, loss_att, loss_ctc = model(feats, x_mask, labels, y_mask, _y, _ylens)
            loss_att = loss_att if loss_att is not None else 0
            loss_ctc = loss_ctc if loss_ctc is not None else 0


            dev_loss.append(loss.item())
            ed_time = time.time()

            total_uttr_dev += num_uttr
            dev_log = "propossed=[%d], loss=%f, duration_batch=%f" % \
                        (total_uttr_dev, np.mean(dev_loss), ed_time - st_time)
            writelog(cur_log, train_log)
        dev_log = "iter=%d, lr=%f, dev loss=%f (cur: loss: %f, loss_att: %f, loss_ctc: %f)" % (cur_iter, cur_lr, np.mean(dev_loss), loss.item(), loss_att.item(), loss_ctc.item())
        writelog(cur_log, dev_log)
        writelog(cur_log, '------------------------------------------------------------')

        if cur_iter >= 2:
            scheduler.step(np.mean(dev_loss))
        
        save_checkpoint("iter%d.pth"%cur_iter, save_dir, cur_iter, model, optimizer, scheduler)

        # logging
        datetime = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        msg = "%s,%d,%f,%f,%f"%(datetime, cur_iter, cur_lr, np.mean(train_loss), np.mean(dev_loss))
        log_history(save_dir, msg)

        cur_iter += 1

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='las train')
    parser.add_argument('--save_dir', type=str, default='results/aishell/mlp')
    parser.add_argument('--init_model', type=str, default='')
    parser.add_argument('--conf_path', type=str, default='')
    parser.add_argument('--n_works', type=int, default=8)
    parser.add_argument('--train_data_file', type=str, default='')
    parser.add_argument('--dev_data_file', type=str, default='')
    parser.add_argument('--vocab', type=str, default='')
    parser.add_argument('--cmvn_file', type=str, default='')
    args = parser.parse_args()

    with open(args.conf_path, 'r', encoding='utf-8') as f:
        conf = yaml.load(f, Loader=yaml.FullLoader)
    
    main(args, conf)


    
