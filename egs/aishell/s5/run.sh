#!/usr/bin/env bash

stage=0
stop_stage=0
gpu=1

corpus=aishell1
# data


# vocab
unit=char
# asr conf
conf=conf/las.yaml
asr_init=

# lm conf
lm_conf=

# path to save the model
model=results/${corpus}

# path to the model directory to resume
resume=
lm_resume=

save_dir=results/aishell/1108_mlp

# path to save preprocessed data
# Select the downloaded data
download_data=/home/Liangcd/data/aishell
# aishell_audio_dir=/home/Liangcd/data/aishell/data_aishell/wav
# aishell_text=/home/Liangcd/data/aishell/data_aishell/transcript/aishell_transcript_v0.8.txt
export data=corpus/${corpus}


. ./path.sh
. parse_options.sh 

set -e 
set -u 
set -o pipefail

train_set=train 
dev_set=dev 
test_set="test"

if [ ${stage} -le -1 ] && [ ${stop_stage} -ge -1 ]; then
    echo "stage -1: Data Download"
    mkdir -p ${data}
    local/download_and_untar.sh ${data} "www.openslr.org/resources/33" data_aishell
    local/download_and_untar.sh ${data} "www.openslr.org/resources/33" resource_aishell
fi


if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
   
    echo "++++++Data Preparation (stage:0)++++++"
    if [ -z "${download_data}" ]; then
        local/aishell_data_prep.sh ${data}/data_aishell/wav ${data}/data_aishell/transcript
    else 
        local/aishell_data_prep.sh ${download_data}/data_aishell/wav ${download_data}/data_aishell/transcript
    fi 
    # remove space in text
    for x in train dev test; do
        cp ${data}/${x}/text ${data}/${x}/text.org
        paste -d " " <(cut -f 1 -d" " ${data}/${x}/text.org) <(cut -f 2- -d" " ${data}/${x}/text.org | tr -d " ") \
            > ${data}/${x}/text
        rm ${data}/${x}/text.org
    done

    echo "Finish data preparation (stage: 0)."
fi


if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    echo "++++++global cmvn (stage:1)++++++"
    
    # feature extraction in training
    # global_cmvn
    compute_cmvn.py --train_config ${conf} \
                    --in_scp ${data}/train/wav.scp \
                    --out_cmvn ${data}/train/global_cmvn
fi

dict=${data}/dict/vocab.txt; mkdir -p ${data}/dict
if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    echo "++++++Dataset preparation (stage:1)++++++"
    # vocab
    make_vocab.sh --unit ${unit} --speed_perturb false \
            ${data} ${dict} ${data}/train/text || exit 1;
        
    echo "Making dataset tsv files for ASR ... (online)"
    mkdir -p $data/dataset
    for x in train dev test; do
        dump_dir=$data/$x
        # echo dump_dir $dump_dir
        make_dataset.sh --feat ${dump_dir}/wav.scp --unit ${unit} --feat_online true \
            ${dump_dir} $dict > $data/dataset/${x}.tsv || exit 1;
    done
fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    echo "++++++train ASR+++++++"
    CUDA_VISIBLE_DEVICES=1 python ${SEE_ROOT}/src/train.py \
            --save_dir ${save_dir} \
            --conf_path ${conf} \
            --n_works 8 \
            --train_data_file ${data}/dataset/train.tsv \
            --dev_data_file ${data}/dataset/dev.tsv \
            --vocab ${dict} \
            --cmvn_file ${data}/train/global_cmvn

fi 

if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
    echo "++++++test ASR++++++"
    CUDA_VISIBLE_DEVICES=1 python ${SEE_ROOT}/src/test.py \
            --save_dir ${save_dir} \
            --conf_path ${conf} \
            --test_data_file ${data}/dataset/test.tsv \
            --vocab ${dict} \
            --test_iter 99 \
            --cmvn_file ${data}/train/global_cmvn
fi 

export_model=${save_dir}/iter99.pth 
if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
    echo "++++++ export model ++++++"
    python ${SEE_ROOT}/src/export_jit.py \
            --conf_path ${conf} \
            --vocab ${dict} \
            --cmvn_file ${data}/train/global_cmvn \
            --checkpoint ${export_model} \
            --output_file ${save_dir}/final.pt \
            --output_quant_file ${save_dir}/final_quant.pt 

fi 


