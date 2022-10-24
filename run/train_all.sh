# here configs has been set to different seeds
ROOT=run/official_configs
## model
CFG=($(find $ROOT -name "*.py"))
# device
GPUS=[0]

# here exhibit how training is done on one card with one job iteratively
for cfg in ${CFG[@]}; do
  echo $cfg
  python run/train.py --cfg=$CFG --gpu_ids=$GPUS
done
