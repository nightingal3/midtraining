SRC=/work/hdd/bfcu/mliu7/pretrained_chkpts/pythia_1b_128b_fixed
DST=/work/hdd/bfcu/mliu7/pretrained_chkpts/pythia_1b_128b_fixed_midtrain_dclm_from_40k

mkdir -p $DST
for i in $(seq 2000 2000 40000); do
  pad=$(printf "%08d" $i)
  ln -s $SRC/step-$pad    $DST/step-$pad
done