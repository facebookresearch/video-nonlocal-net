
# create lmdb for FCN testing
# each row in lmdb: video_file_name, label, start_frame, spatial position
python create_video_lmdb_test_multicrop.py --dataset_dir ../../data/lmdb/kinetics_lmdb_multicrop/test  --list_file vallist.txt

# create lmdb for FCN testing, adding flipping
mkdir ../../data/lmdb/kinetics_lmdb_flipcrop
python create_video_lmdb_test_flipcrop.py --dataset_dir ../../data/lmdb/kinetics_lmdb_flipcrop/test  --list_file vallist.txt
