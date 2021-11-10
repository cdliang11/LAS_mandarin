#!/bin/bash
# split the wav scp, calculate duration and merge

# ref: https://github.com/wenet-e2e/wenet
nj=4

. parse_options.sh;


inscp=$1
outscp=$2

# echo $inscp
data=$(dirname ${inscp})
# echo $data
# data=corpus/aishell1_online/raw_wav/train

if [ $# -eq 3 ]; then
  logdir=$3
else
  logdir=${data}/log
fi

mkdir -p ${logdir}

rm -f $logdir/wav_*.slice
rm -f $logdir/wav_*.shape
split --additional-suffix .slice -d -n l/$nj $inscp $logdir/wav_ || exit 1;

for slice in ` ls $logdir/wav_*.slice `
do
{
    name=` basename -s .slice $slice `
    wav2dur.py $slice $logdir/$name.shape 1>$logdir/$name.log
} &
done
wait
cat $logdir/wav_*.shape > $outscp || exit 1;
