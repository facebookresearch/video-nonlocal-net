# CUDA_VISIBLE_DEVICES=1,2,3
python ../tools/test_net_video.py \
--config_file ../configs/DBG_kinetics_resnet_8gpu_c2d_nonlocal_400k.yaml \
NUM_GPUS 4 \
TRAIN.BATCH_SIZE 32 \
TEST.BATCH_SIZE 32 \
TEST.PARAMS_FILE ../data/checkpoints/run_i3d_nlnet_400k/checkpoints/c2_model_iter400000.pkl \
VIDEO_DECODER_THREADS 5 \
NONLOCAL.CONV3_NONLOCAL True \
NONLOCAL.CONV4_NONLOCAL True \
MODEL.VIDEO_ARC_CHOICE 2 \
TRAIN.DROPOUT_RATE 0.5 \
DATADIR ../data/lmdb/kinetics_lmdb_multicrop/ \
FILENAME_GT ../process_data/kinetics/vallist.txt \
TEST.TEST_FULLY_CONV True
