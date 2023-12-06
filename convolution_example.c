/*
 * convolution_example.c is an example script that does the same thing
 * as traditional_example.c but with a convolutional architecture.
 * Should hopefully work at a much higher accuracy than the traditional example.
 */

//Standard library headers
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

//GSL headers
#include <gsl/gsl_math.h>
#include <gsl/gsl_randist.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_cdf.h>

//My headers
#include "structs.h"
#include "netex.h"
#include "files.h"

int main() {
	//Declare variables
		//Iteration variables
	int g, h, i, j, k;
		//Ints for meta-parameters
	int epochs, blocks, block_count, test_count;
		//Ints for label and size of output layer
	int label, out_size;
	
	//Architecture setting constants

	//Training/testing loop constants
	epochs = 50;
	blocks = 1000;
	block_count = 5;
	test_count = 1000;

	/*
	 * DEBUG.
	 * BIT1 = Printout the empty architecture after initialization
	 * BIT2 = Printout the expected output vector
	 * BIT3 = Print output after feedforward
	 * BIT4 = Printout the gradient of output layer after backprop
	 * BIT5 = Printout the average gradient of output layer
	 * BIT6
	 * BIT7 = Load in settings file
	 * BIT8 = Enable training
	 * BIT9 = Save settings after each epoch
	 */
	const int debug = BIT1 | BIT8 | BIT9;

	//Set up initialization settings
		//User should put their settings in these arrays. This will be
		//converted into an initialization structure, which is then converted
		//into a fleshed-out architecture.
	char types[] = {'c', 'c', 'p', 't'};
	int neurons[] = {28, 24, 12, 10};
	int depths[] = {1, 3, 3, 1};
	//Because input & trad layers have to rfield, first and last vals are arbitrary
	int rfields[] = {0, 5, 3, 0};

   		//Get layer count from array	
	const int layer_count = sizeof(types) / sizeof(char); 

		//Declare network and layer init
	conv_init network;
	conv_layer_init lays[layer_count];

		//Get layer count
	network.layers = layer_count;
		//Point network layer array pointer to layers
	network.layer_data = &lays[0];

	//Initialize init 
	for (i = 0; i < layer_count; i++) {
		lays[i].size1 = neurons[i];
		lays[i].size2 = neurons[i];

		lays[i].type = types[i];
		
		lays[i].depth = depths[i];

		lays[i].rfield = rfields[i];
	}

	//Get size of output layer
	out_size = lays[layer_count - 1].size1;

	//Get vector array of correct values
	gsl_vector *corrects[10];
	gsl_vector *correct;

	//Fill out corrects array
	for (i = 0; i < out_size; i++) {
		corrects[i] = expected_out(i, out_size);
	}

	//Empty input vector, input goes here before the net
	gsl_vector *in_vect = gsl_vector_calloc(lays[0].size1 * lays[0].size2 + 1);

	//Init network
	conv_layer *net = conv_architecture(&network);

	//Debug 1: Printout architecture info
	if (debug & BIT1) {
		for (i = 0; i < layer_count; i++) {
			printf("Layer: %d\n", i + 1);

			printf("\tType: %c\n", (net) -> type);
		}
	}
}
