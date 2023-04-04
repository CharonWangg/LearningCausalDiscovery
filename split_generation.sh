WINDOWS=(window_1536_1792 window_1792_2048 window_2048_2304 window_2304_2560 window_2560_2816 window_2816_3072 window_3072_3328 window_3328_3584 window_3584_3840 window_3840_4096 window_4096_4352 window_4352_4608 window_4608_4864 window_4864_5120 window_5120_5376 window_5376_5632 window_5632_5888 window_5888_6144 window_6144_6400 window_6400_6656 window_6656_6912 window_6912_7168 window_7168_7424 window_7424_7680 window_7680_7936 window_7936_8192)
JOBS=4

parallel --progress --linebuffer --jobs=$JOBS --delay=120 \
  python nmos_simulation/causal_simulation/split_generation.py --game=DonkeyKong --split={} ::: ${WINDOWS[@]}