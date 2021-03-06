# LAS_mandarin
Listen, attend and spell Model for Mandarin

## Install
- Python3
- Pytorch 1.8+
- torchaudio

## Usage
1. `cd egs/aishell/s5/`，and modify aishell data path to your path in `run.sh`
2. `bash run.sh`

## Workflow of run.sh
- stage -1: Data Download
- stage 0: Data Preparation
- stage 1: Global cmvn
- stage 2: Dataset preparation
- stage 3: Train ASR
- stage 4: Test ASR

## Results

| Model                                                   | wer (%) |
| ------------------------------------------------------- | ------- |
| Listen, Attend and Spell （no LM）[1]  ctcprefix_beam10 | 10.26   |
| +  joint CTC/attention  ctc0                            | 9.10    |
| + ctc0.3                                                | 8.93    |


## Reference：

1. [Listen, Attend and Spell](https://arxiv.org/abs/1508.01211v2), W Chan et al.
2. [wenet](https://github.com/wenet-e2e/wenet)
3. [neural_sp](https://github.com/hirofumi0810/neural_sp)

## TODO
- [x] wfst decode (py)
- [x] export model
- [ ] decode (libtorch)