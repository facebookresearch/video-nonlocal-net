CHECKPOINT_DIR=../data/checkpoints/run_c2d_baseline_400k_32f
mkdir ${CHECKPOINT_DIR}

python ../tools/train_net_video.py \
--config_file ../configs/DBG_kinetics_resnet_8gpu_c2d_nonlocal_400k.yaml \
TRAIN.PARAMS_FILE ../data/pretrained_model/r50_pretrain_c2_model_iter450450_clean.pkl \
VIDEO_DECODER_THREADS 5 \
NONLOCAL.CONV3_NONLOCAL False \
NONLOCAL.CONV4_NONLOCAL False \
TRAIN.VIDEO_LENGTH 32 \
TRAIN.SAMPLE_RATE 2 \
TEST.VIDEO_LENGTH 32 \
TEST.SAMPLE_RATE 2 \
MODEL.MODEL_NAME resnet_video_org \
MODEL.VIDEO_ARC_CHOICE 1 \
TRAIN.DROPOUT_RATE 0.5 \
CHECKPOINT.DIR ${CHECKPOINT_DIR} \
DATADIR ../data/lmdb/kinetics_lmdb_multicrop/ \
FILENAME_GT ../process_data/kinetics/vallist.txt \
2>&1 | tee ${CHECKPOINT_DIR}/log.txt
