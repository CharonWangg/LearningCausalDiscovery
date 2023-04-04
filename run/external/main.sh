NETSIM_MODELS=(sru eSRU_2LF)
NETSIM_DSID=(1 2 3 8 10 11 12 13 14 15 16 17 18 21)
DREAM3_MODELS=(sru eSRU_1LF)
DREAM3_DSID=(2 4 5)
JOBS=3

# regular test on netsim
# sru
#CUDA_VISIBLE_DEVICES=2 parallel --linebuffer --jobs=$JOBS python run/external/SRU_for_GCI/main.py --dataset netsim --dsid={1}  \
#  --model=sru --T 200 --F 0 --nepochs 2000 --mu1 0.464159 --mu2 0.1 --mu3 0.0 \
#  --lr 0.001 --joblog .cache/netsim/logs/regular_ --condition=regular --percentage_start=0.8 --percentage_end=1.0 \
#  ::: ${NETSIM_DSID[@]}

# esru
#CUDA_VISIBLE_DEVICES=2 parallel --linebuffer --jobs=$JOBS python run/external/SRU_for_GCI/main.py --dataset netsim --dsid={1}  \
#  --model=eSRU_2LF --T 200 --F 0 --nepochs 2000 --mu1 0.232 --mu2 0.1 --mu3 0.005 \
#  --lr 0.001 --joblog .cache/netsim/logs/regular_ --condition=regular --percentage_start=0.8 --percentage_end=1.0 \
#  ::: ${NETSIM_DSID[@]}

# noise test on netsim
# sru
#CUDA_VISIBLE_DEVICES=2 parallel --linebuffer --jobs=$JOBS python run/external/SRU_for_GCI/main.py --dataset netsim --dsid={1}  \
#  --model=sru --T 200 --F 0 --nepochs 2000 --mu1 0.464159 --mu2 0.1 --mu3 0.0 \
#  --lr 0.001 --joblog .cache/netsim/logs/noise_ --condition=noise --percentage_start=0.8 --percentage_end=1.0 \
#  ::: ${NETSIM_DSID[@]}
#
## esru
#CUDA_VISIBLE_DEVICES=2 parallel --linebuffer --jobs=$JOBS python run/external/SRU_for_GCI/main.py --dataset netsim --dsid={1}  \
#  --model=eSRU_2LF --T 200 --F 0 --nepochs 2000 --mu1 0.232 --mu2 0.1 --mu3 0.005 \
#  --lr 0.001 --joblog .cache/netsim/logs/noise_ --condition=noise --percentage_start=0.8 --percentage_end=1.0 \
#  ::: ${NETSIM_DSID[@]}

## sru test on dream3
CUDA_VISIBLE_DEVICES=2 parallel --linebuffer --jobs=$JOBS python run/external/SRU_for_GCI/main.py --dataset gene --dsid={1} --model=sru \
  --n 100 --T 966 --F 0 --nepochs 1000 --mu1 0.2 --mu2 0.031623 --mu3 0.0 \
  --lr 0.005 --joblog .cache/dream3/logs/test_ ::: ${DREAM3_DSID[@]}

# esru test on dream3
CUDA_VISIBLE_DEVICES=2 parallel --linebuffer --jobs=$JOBS python run/external/SRU_for_GCI/main.py --dataset gene --dsid={1} --model=eSRU_1LF \
  --n 100 --T 966 --F 0 --nepochs 2000 --mu1 0.1 --mu2 0.031623 --mu3 1.0 \
  --lr 0.001 --joblog .cache/dream3/logs/test_ ::: ${DREAM3_DSID[@]}