python tools/run_net.py \
		   --cfg $1 \
		   --init_method 'tcp://10.215.241.105:7017' \
		   --num_shards 2 \
		   --shard_id 1 \
		   NUM_GPUS $2 \
		   TRAIN.BATCH_SIZE $((16*$2)) \
		   DATA.PATH_TO_DATA_DIR /home/keyu/keyu/data/videos
