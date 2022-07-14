HLS_INCLUDE = /opt/Xilinx/Vitis_HLS/2021.2/include/

all:
	g++ -fPIC ap.cpp -c -I${HLS_INCLUDE}
	gcc -c -Wall -Werror -fpic cmx.c -I /usr/include/python3.7
	g++ -shared -o libcmx.so cmx.o ap.o
	python3 build-cffi.py

clean:
	rm -f *.o *.so
	rm -f cmx_cffi.c
