python tools/run_net.py \
		   --cfg $1 \
		   NUM_GPUS $2 \
		   TRAIN.BATCH_SIZE $((16*$2)) \
		   DATA.PATH_TO_DATA_DIR /home/layer6/chundi/epic/rgb/
