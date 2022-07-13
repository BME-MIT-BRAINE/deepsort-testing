make -C cffi_matrix/ clean all

export PYTHONPATH=${PWD}/cffi_matrix/
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${PWD}/cffi_matrix/

python object_tracker.py --video ./data/video/test.mp4 --model yolov4 --load_checkpoint "$@"
