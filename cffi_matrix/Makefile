all:
	gcc -c -Wall -Werror -fpic cmx.c -I /usr/include/python3.7
	gcc -shared -o libcmx.so cmx.o
	python3 build-cffi.py

clean:
	rm -f *.o *.so
	rm -f cmx_cffi.c
