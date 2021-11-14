
from __future__ import print_function

import argparse
import os
from typing_extensions import ParamSpec

import torch
import yaml
from utils import load_cmvn, GlobalCMVN
import build_model


def main(args):
    with open(args.conf_path, 'r', encoding='utf-8') as f:
        configs = yaml.load(f, Loader=yaml.FullLoader)
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
        print('no global cmvn')
    
    model = build_model.LAS(feat_dim=80, vocab=vocab_size, global_cmvn=global_cmvn)
    print(model)
    num_params = sum(p.numel() for p in model.parameters())
    print("The number of model params: {}".format(num_params))
    

    if args.checkpoint:
        print('load init_model from %s' % args.checkpoint)
        info = torch.load(args.checkpoint)
        model.load_state_dict(info['weights'])
    
    # TODO: Export jit torch script model
    script_model = torch.jit.script(model)
    script_model.save(args.output_file)
    print('Export model successfully, see {}.'.format(args.output_file))

    # TODO: Export quantized jit torch script model
    if args.output_quant_file:
        quantized_mode = torch.quantization.quantize_dynamic(
            model, {torch.nn.Linear}, dtype=torch.qint8
        )
        print(quantized_mode)
        num_params = sum(p.numel() for p in quantized_mode.parameters())
        print("The number of model params: {}".format(num_params))

        script_quant_model = torch.jit.script(quantized_mode)
        script_quant_model.save(args.output_quant_file)
        print('Export quantized model successfully, see {}.'.format(
            args.output_quant_file
        ))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='export pytorch model')
    parser.add_argument('--conf_path', required=True, help='config file')
    parser.add_argument('--vocab', required=True, help='vocab file')
    parser.add_argument('--cmvn_file', required=True, help='global cmvn file')
    parser.add_argument('--checkpoint', required=True, help='checkpoint model')
    parser.add_argument('--output_file', required=True, help='output file')
    parser.add_argument('--output_quant_file', default=None, help='output quantized model file')
    args = parser.parse_args()
    ## no need gpu for model export
    # os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    main(args)