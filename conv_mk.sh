gcc -Wall -I/usr/local/include -c conv_net.c netex.c files.c convolution_example.c
gcc -L/usr/local/lib conv_net.o netex.o files.o convolution_example.o -lgsl -lgslcblas -lm
