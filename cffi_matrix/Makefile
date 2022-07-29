HLS_INCLUDE = $(XILINX_HLS)/include

SRC_DIR = deepsort-accel/src

check-hls:
ifeq ($(XILINX_HLS),)
	$(error ERROR: 'XILINX_HLS' variable not set. Source settings script, or set manually to <install path>/Vitis_HLS/<version>/)
endif

all: check-hls
	g++ -fPIC $(SRC_DIR)/deepsort_cmx.cpp -c -I${HLS_INCLUDE}
	gcc -c -Wall -Werror -fpic  $(SRC_DIR)/deepsort_cmx_c_wrapper.c -I /usr/include/python3.7
	g++ -shared -o libcmx.so deepsort_cmx_c_wrapper.o deepsort_cmx.o
	python3 build-cffi.py

clean:
	rm -f *.o *.so
	rm -f cmx_cffi.c
