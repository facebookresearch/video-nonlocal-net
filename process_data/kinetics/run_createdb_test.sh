# create lmdb for FCN testing
# each row in lmdb: video_file_name, label, start_frame
mkdir ../../data/lmdb/kinetics_lmdb_singlecrop
python create_video_lmdb_test.py --dataset_dir ../../data/lmdb/kinetics_lmdb_singlecrop/test  --list_file vallist.txt
