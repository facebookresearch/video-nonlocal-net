
# create training lmdb:
# each row in lmdb: video_file_name, label
mkdir ../../data/lmdb
mkdir ../../data/lmdb/kinetics_lmdb_multicrop
python create_video_lmdb.py --dataset_dir ../../data/lmdb/kinetics_lmdb_multicrop/train  --list_file trainlist_shuffle_rep.txt

# create val lmdb:
python create_video_lmdb.py --dataset_dir ../../data/lmdb/kinetics_lmdb_multicrop/val  --list_file vallist.txt
