CHECKPOINT_DIR=../data/checkpoints/run_c2d_nlnet_400k
mkdir ${CHECKPOINT_DIR}

python ../tools/train_net_video.py \
--config_file ../configs/DBG_kinetics_resnet_8gpu_c2d_nonlocal_400k.yaml \
TRAIN.PARAMS_FILE ../data/pretrained_model/r50_pretrain_c2_model_iter450450_clean.pkl \
VIDEO_DECODER_THREADS 5 \
NONLOCAL.CONV3_NONLOCAL True \
NONLOCAL.CONV4_NONLOCAL True \
MODEL.VIDEO_ARC_CHOICE 1 \
TRAIN.DROPOUT_RATE 0.5 \
CHECKPOINT.DIR ${CHECKPOINT_DIR} \
DATADIR ../data/lmdb/kinetics_lmdb_multicrop/ \
FILENAME_GT ../process_data/kinetics/vallist.txt \
2>&1 | tee ${CHECKPOINT_DIR}/log.txt
