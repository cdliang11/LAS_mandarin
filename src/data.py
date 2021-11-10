# Copyright (c) 2020 Mobvoi Inc. (authors: Binbin Zhang, Chao Yang)
# Copyright (c) 2021 Jinsong Pan
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.



import torch
import numpy as np
import random
from torch._C import _is_tracing
import torchaudio
import torchaudio.compliance.kaldi as kaldi
import torchaudio.sox_effects as sox_effects
from torch.utils.data import DataLoader, Dataset
import codecs
import pandas as pd
import argparse
import yaml
import copy
from torch.nn.utils.rnn import pad_sequence
from PIL import Image
from PIL.Image import BICUBIC

# pad
IGNORE_ID = -1

def _spec_augmentation(x,
                       warp_for_time=False,
                       num_t_mask=2,
                       num_f_mask=2,
                       max_t=50,
                       max_f=10,
                       max_w=80):
    """ Deep copy x and do spec augmentation then return it

    Args:
        x: input feature, T * F 2D
        num_t_mask: number of time mask to apply
        num_f_mask: number of freq mask to apply
        max_t: max width of time mask
        max_f: max width of freq mask
        max_w: max width of time warp

    Returns:
        augmented feature
    """
    y = np.copy(x)
    max_frames = y.shape[0]
    max_freq = y.shape[1]

    # time warp
    if warp_for_time and max_frames > max_w * 2:
        center = random.randrange(max_w, max_frames - max_w)
        warped = random.randrange(center - max_w, center + max_w) + 1

        left = Image.fromarray(x[:center]).resize((max_freq, warped), BICUBIC)
        right = Image.fromarray(x[center:]).resize(
            (max_freq, max_frames - warped), BICUBIC)
        y = np.concatenate((left, right), 0)
    # time mask
    for i in range(num_t_mask):
        start = random.randint(0, max_frames - 1)
        length = random.randint(1, max_t)
        end = min(max_frames, start + length)
        y[start:end, :] = 0
    # freq mask
    for i in range(num_f_mask):
        start = random.randint(0, max_freq - 1)
        length = random.randint(1, max_f)
        end = min(max_freq, start + length)
        y[:, start:end] = 0
    return y

def _load_wav_with_speed(wav_file, speed):
    """ 
    ref: https://github.com/wenet-e2e/wenet
    Load the wave from file and apply speed perpturbation

    Args:
        wav_file: input feature, T * F 2D

    Returns:
        augmented feature
    """
    if speed == 1.0:
        wav, sr = torchaudio.load(wav_file)
    else:
        sample_rate = torchaudio.backend.sox_io_backend.info(
            wav_file).sample_rate
        # get torchaudio version
        ta_no = torchaudio.__version__.split(".")
        ta_version = 100 * int(ta_no[0]) + 10 * int(ta_no[1])

        if ta_version < 80:
            # Note: deprecated in torchaudio>=0.8.0
            E = sox_effects.SoxEffectsChain()
            E.append_effect_to_chain('speed', speed)
            E.append_effect_to_chain("rate", sample_rate)
            E.set_input_file(wav_file)
            wav, sr = E.sox_build_flow_effects()
        else:
            # Note: enable in torchaudio>=0.8.0
            wav, sr = sox_effects.apply_effects_file(
                wav_file,
                [['speed', str(speed)], ['rate', str(sample_rate)]])

    return wav, sr

def _feature_extraction(batch, speed_perturb, feature_conf):
    """
    fbank feature
    Args:
        wav (list): the path of raw wav
        speed_perturb (bool): is or not to use speed pertubation
        feature_conf (dict): the conf of fbank extraction
    Returns:
        (uttrs, feats, labels)
    """
    uttrs = []
    feats = []
    xlens = []
    ylens = []

    if speed_perturb:
        speeds = [0.9, 1.0, 1.1]
        weights = [1, 1, 1]
        speed = random.choices(speeds, weights, k=1)[0]
    
    for i, x in enumerate(batch):
        try:
            wav = x[1] # wav_path

            sample_rate = torchaudio.backend.sox_io_backend.info(wav).sample_rate
            # if 'resample' in feature_conf:
            #     resample_rate = feature_conf['resample']
            # else:
            #     resample_rate = sample_rate
            resample_rate = getattr(feature_conf, 'resample', sample_rate)
            if speed_perturb:
                wavform, sample_rate = _load_wav_with_speed(wav, speed)
            else:
                wavform, sample_rate = torchaudio.load(wav)
            
            wavform = wavform * (1 << 15)

            if resample_rate != sample_rate:
                wavform = torchaudio.transforms.Resample(
                    orig_freq=sample_rate, new_freq=resample_rate
                )(wavform)

            feat = torchaudio.compliance.kaldi.fbank(
                waveform=wavform,
                num_mel_bins=feature_conf['mel_bins'],
                frame_length=feature_conf['frame_length'],
                frame_shift=feature_conf['frame_shift'],
                energy_floor=0.0,
                sample_frequency=resample_rate
            )

            feat = feat.detach().numpy() # [frame, adim]
            feats.append(feat)
            uttrs.append(x[0])
            xlens.append(feat.shape[0])
            ylens.append(x[3])
           
        except (Exception) as e:
            print(e)
            pass 
    # sort 
    _index = np.argsort(xlens)[::-1]
    sorted_uttrs = [uttrs[i] for i in _index]
    sorted_feats = [feats[i] for i in _index]
    # token_id
    labels = [['4233'] + x[2].split() + ['4234'] for x in batch] # add eos
    # print(labels)
    # assert False
    labels = [np.fromiter(map(int, x), dtype=np.int32) for x in labels]
    sorted_labels = [labels[i] for i in _index]
    sorted_ylens = [int(ylens[i])+2 for i in _index]
    return sorted_uttrs, sorted_feats, sorted_labels, sorted_ylens

    

class collatefunc(object):
    """
    collate function for audioDataset
    """

    def __init__(self,
                speed_perturb=False,
                spec_aug=False,
                spec_aug_conf=None,
                feature_conf=None):
        self.speed_perturb = speed_perturb
        self.spec_aug = spec_aug
        self.spec_aug_conf = spec_aug_conf
        self.feature_conf = feature_conf

    def __call__(self, batch):
        assert len(batch) == 1
        
        # feature extraction
        uttrs, feats, labels, ylens = _feature_extraction(batch[0], self.speed_perturb, self.feature_conf)
        
        is_training = True
        if labels is None:
            is_training = False
        
        if self.spec_aug:
            feats = [_spec_augmentation(x, **self.spec_aug_conf) for x in feats]
        
        xlens = torch.from_numpy(
            np.array([feat.shape[0] for feat in feats], dtype=np.int32)
        )
        # padding
        if len(feats) > 0:
            feats_pad = pad_sequence([torch.from_numpy(feat).float() for feat in feats], True, 0)
        else:
            feats_pad = torch.Tensor(feats)
        
        if is_training:
            # print(labels)
            # assert False
            _ylens = torch.from_numpy(
                np.array([y.shape[0] for y in labels], dtype=np.int32)
            )
            if len(_ylens) > 0:
                labels_pad = pad_sequence([torch.from_numpy(y).int() for y in labels], True, IGNORE_ID)
            else:
                labels_pad = torch.Tensor(labels)
        else:
            labels_pad = None
            _ylens = None
        return uttrs, feats_pad, labels_pad, xlens, _ylens



class AudioDataset(Dataset):
    def __init__(self,
                data_file,
                max_length=10240,
                min_length=0,
                token_max_length=200,
                token_min_length=1,
                dynamic_batch=False,
                batch_size=1,
                max_frames_in_batch=0,
                is_test=False):
        """
        Dataset for loading audio data.
        """
        super(Dataset, self).__init__()
        data = []

        # load dataset tsv file
        chunk = pd.read_csv(data_file, encoding='utf-8',
                            delimiter='\t', chunksize=1000000)
        df = pd.concat(chunk)
        df = df.loc[:, ['utt_id', 'speaker', 'feat_path',
                        'xlen', 'xdim', 'text', 'token_id', 'ylen', 'ydim']]
        if not is_test:
            df.sort_values(by=['xlen']) # ascending order
        n_uttr = len(df)
        if is_test:
            df = df[df.apply(lambda x : x['ylen'] > 0, axis=1)]
            print(f'Removed {n_uttr - len(df)} empty utterances')
        else:
            # remove too lang or too short uttr 
            df = df[df.apply(lambda x : min_length <= x [
                'xlen'] <= max_length, axis=1)]
            df = df[df.apply(lambda x : token_min_length <= x[
                'ylen'] <= token_max_length, axis=1)]
        # remove empty utterances
        df = df[df.apply(lambda x : x['ylen'] > 0, axis=1)]
        print(f'Removed {n_uttr - len(df)} utterances (threshold)')

        self.minibatch = []
        num_data = len(df)
        print('batch_size', batch_size)
        if dynamic_batch :
            assert max_frames_in_batch > 0
            self.minibatch.append([])
            num_frames_in_batch = 0
            cnt = 0 # 
            for i in range(num_data):
                xlen = df.iloc[i]['xlen']
                num_frames_in_batch += xlen
                cnt += 1
                if num_frames_in_batch > max_frames_in_batch or cnt > batch_size:
                    self.minibatch.append([])
                    num_frames_in_batch = xlen
                    cnt = 1
                
                self.minibatch[-1].append((df.iloc[i]['utt_id'], df.iloc[i]['feat_path'], df.iloc[i]['token_id'], df.iloc[i]['ylen']))
        else:
            # static batch size
            cur = 0
            # print(df.iloc[0])
            while cur < num_data:
                end = min(cur+batch_size, num_data)
                item = []
                for i in range(cur, end):
                    item.append((df.iloc[i]['utt_id'], df.iloc[i]['feat_path'], df.iloc[i]['token_id'], df.iloc[i]['ylen']))
                self.minibatch.append(item)
                cur = end 
    def __len__(self):
        return len(self.minibatch)
    
    def __getitem__(self, index):
        return self.minibatch[index]
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--type', help='config file')
    parser.add_argument('--config_file', help='')
    parser.add_argument('--data_file', help='')
    args = parser.parse_args()

    with open(args.config_file, 'r') as f:
        configs = yaml.load(f, Loader=yaml.FullLoader)
    
    # init dataset
    data_conf = copy.copy(configs['data'])
    if args.type == 'raw_wav':
        raw_wav = True
    else:
        raw_wav = False

    collate_func = collatefunc(**data_conf['collate_fn'])
    print(data_conf)
    dataset_conf = data_conf.get('dataset', {})
    dataset = AudioDataset(args.data_file, **dataset_conf, is_test=False)

    data_loader = DataLoader(dataset,
                            batch_size=1,
                            shuffle=True,
                            sampler=None,
                            num_workers=0,
                            collate_fn=collate_func)
    
    for i, batch in enumerate(data_loader):
        # print(batch)
        pass
    



        