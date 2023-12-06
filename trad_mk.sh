gcc -Wall -I/usr/local/include -c netex.c files.c traditional_example.c
gcc -L/usr/local/lib netex.o files.o example.o -lgsl -lgslcblas -lm
