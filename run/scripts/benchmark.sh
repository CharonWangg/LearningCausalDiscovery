NETSIM_DSID=(1 2 3 8 10 11 12 13 14 15 16 17 18 21 22 23 24)
DREAM3_DSID=(Ecoli2 Yeast2 Yeast3)
SAVE_DIR=.cache/sim_data/normalized
DEVICE=2
JOBS=3

# regular test on netsim
#CUDA_VISIBLE_DEVICES=$DEVICE parallel --linebuffer --jobs=$JOBS python run/scripts/netsim_auc_test.py --root_dir=$SAVE_DIR --dataset_id={1} \
#  --condition=default ::: ${NETSIM_DSID[@]}


#for dsid in ${NETSIM_DSID[@]}; do
#  CUDA_VISIBLE_DEVICES=$DEVICE python run/scripts/netsim_auc_test.py --dataset_id=$dsid --condition=default
#done

# noise test on netsim
#CUDA_VISIBLE_DEVICES=$DEVICE parallel --linebuffer --jobs=$JOBS python run/scripts/netsim_auc_test.py --dataset_id={1} \
#  --condition=noise ::: ${NETSIM_DSID[@]}
#
### regular test on dream3
CUDA_VISIBLE_DEVICES=$DEVICE parallel --linebuffer --jobs=$JOBS python run/scripts/dream3_auc_test.py --root_dir=$SAVE_DIR --dataset_id={1} \
  ::: ${DREAM3_DSID[@]}