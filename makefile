log : example.out
	./example.out

example.out : traditional_example.o netex.o files.o
	cc -Wall -L/usr/local/lib -o example.out traditional_example.o netex.o files.o -lgsl -lgslcblas -lm

netex.o : netex.c structs.h netex.h
	cc -Wall -I/usr/local/include -c netex.c

files.o : files.c netex.h structs.h
	cc -Wall -I/usr/local/include -c files.c
