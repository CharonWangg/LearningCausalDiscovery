NETSIM_DSID=(1 2 3 8 10 11 12 13 14 15 16 17 18 21)
DREAM3_DSID=(Ecoli2 Yeast2 Yeast3)
JOBS=12

# regular test on netsim
CUDA_VISIBLE_DEVICES=2 parallel --linebuffer --jobs=$JOBS python run/scripts/netsim_auc_test.py --dataset_id={1}  \
  --condition=default ::: ${NETSIM_DSID[@]}

# noise test on netsim
CUDA_VISIBLE_DEVICES=2 parallel --linebuffer --jobs=$JOBS python run/scripts/netsim_auc_test.py --dataset_id={1} \
  --condition=noise ::: ${NETSIM_DSID[@]}

## regular test on dream3
CUDA_VISIBLE_DEVICES=2 parallel --linebuffer --jobs=$JOBS python run/scripts/dream3_auc_test.py --dataset_id={1} \
  ::: ${DREAM3_DSID[@]}