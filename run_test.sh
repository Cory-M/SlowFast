python tools/run_net.py \
		   --cfg $1 \
		   NUM_GPUS $2 \
		   TRAIN.BATCH_SIZE $((16*$2)) \
		   TRAIN.ENABLE False \
		   DATA.RANDOM_FLIP False \
		   DATA.CVRL_AUG False \
		   TEST.SECTION train \
		   TEST.CHECKPOINT_FILE_PATH /home/layer6/chundi/logs/cvrl_shuffleBN_epic/checkpoints/checkpoint_epoch_00196.pyth \
		   DATA.PATH_TO_DATA_DIR /home/layer6/chundi/epic/rgb/
