
import argparse
from collections import defaultdict
import numpy as np
from tqdm import tqdm

def compute_wer(ref, hyp, normalize=False):
    """Compute Word Error Rate.

        [Reference]
            https://martin-thoma.com/word-error-rate-calculation/
    Args:
        ref (list): words in the reference transcript
        hyp (list): words in the predicted transcript
        normalize (bool, optional): if True, divide by the length of ref
    Returns:
        wer (float): Word Error Rate between ref and hyp
        n_sub (int): the number of substitution
        n_ins (int): the number of insertion
        n_del (int): the number of deletion

    """
    # Initialisation
    d = np.zeros((len(ref) + 1) * (len(hyp) + 1), dtype=np.uint16)
    d = d.reshape((len(ref) + 1, len(hyp) + 1))
    for i in range(len(ref) + 1):
        for j in range(len(hyp) + 1):
            if i == 0:
                d[0][j] = j
            elif j == 0:
                d[i][0] = i

    # Computation
    for i in range(1, len(ref) + 1):
        for j in range(1, len(hyp) + 1):
            if ref[i - 1] == hyp[j - 1]:
                d[i][j] = d[i - 1][j - 1]
            else:
                sub_tmp = d[i - 1][j - 1] + 1
                ins_tmp = d[i][j - 1] + 1
                del_tmp = d[i - 1][j] + 1
                d[i][j] = min(sub_tmp, ins_tmp, del_tmp)

    wer = d[len(ref)][len(hyp)]

    # Find out the manipulation steps
    x = len(ref)
    y = len(hyp)
    error_list = []
    while True:
        if x == 0 and y == 0:
            break
        else:
            if x > 0 and y > 0:
                if d[x][y] == d[x - 1][y - 1] and ref[x - 1] == hyp[y - 1]:
                    error_list.append("C")
                    x = x - 1
                    y = y - 1
                elif d[x][y] == d[x][y - 1] + 1:
                    error_list.append("I")
                    y = y - 1
                elif d[x][y] == d[x - 1][y - 1] + 1:
                    error_list.append("S")
                    x = x - 1
                    y = y - 1
                else:
                    error_list.append("D")
                    x = x - 1
            elif x == 0 and y > 0:
                if d[x][y] == d[x][y - 1] + 1:
                    error_list.append("I")
                    y = y - 1
                else:
                    error_list.append("D")
                    x = x - 1
            elif y == 0 and x > 0:
                error_list.append("D")
                x = x - 1
            else:
                raise ValueError

    n_sub = error_list.count("S")
    n_ins = error_list.count("I")
    n_del = error_list.count("D")
    n_cor = error_list.count("C")

    assert wer == (n_sub + n_ins + n_del) # 
    assert n_cor == (len(ref) - n_sub - n_del)

    if normalize:
        wer /= len(ref)

    return wer * 100, n_sub * 100, n_ins * 100, n_del * 100

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='compute wer')
    parser.add_argument('--ref', type=str, required=True)
    parser.add_argument('--pred', type=str, required=True)
    args = parser.parse_args()

    with open(args.ref, 'r', encoding='utf-8') as f:
        ref = defaultdict(str)
        for line in f.readlines():
            tmp = line.strip().split(' ')
            ref[tmp[0]] = tmp[1]
    
    with open(args.pred, 'r', encoding='utf-8') as f:
        pred = defaultdict(str)
        for line in f.readlines():
            tmp = line.strip().split(' ')
            pred[tmp[0]] = tmp[1]
    
    assert len(ref) == len(pred), 'ref or pred is wrong'

    total_wer = 0
    cnt = 0
    total_sub = 0
    total_ins = 0
    total_del = 0
    for uttr in tqdm(ref.keys()):
        # print(ref[uttr])
        # print(pred[uttr])
        wer, n_sub, n_ins, n_del = compute_wer(ref[uttr], pred[uttr])
        # print(wer, n_sub, n_ins, n_del)
        # assert False
        total_wer += wer    
        total_sub += n_sub
        total_ins += n_ins
        total_del += n_del
        cnt += len(ref[uttr])
    
    print('wer: %f, sub: %f, ins: %f, del: %f\n'%(total_wer/cnt, total_sub/cnt, total_ins/cnt, total_del/cnt))