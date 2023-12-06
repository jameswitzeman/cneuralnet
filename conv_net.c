/*
 * conv_layer.c implements functions for initializing,
 * feedforwarding, and backpropagating a convolutional neural network.
 */

//Standard library
#include <stdlib.h>
#include <stdio.h>
#include <math.h>

//GSL
#include <gsl/gsl_math.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#include <gsl/gsl_cdf.h>

//My Scripts	
#include "netex.h"
#include "structs.h"
#include "conv_net.h"

//Function for initializing a convolutional network
conv_layer *conv_architecture(conv_init *init) {
	//Iteration ints
	int i, j, k;

	//Ints for vector/matrix sizes
	int tsize1, tsize2, tdepth, tfield;

	//Character for holding current layer's type
	char thistype;

	//Get number of layers
	int laycnt = init -> layers;

	//Pointer for holding convolutional array
	conv_layer *lays = malloc(laycnt * sizeof(conv_layer));

	//Pointer for iterating over each initialization layer
	conv_layer_init *thislay, *lastlay;

	//Initialize random number generator for initializing weights
	gsl_rng *prand = gsl_rng_alloc(gsl_rng_mt19937);

	//Seed the RNG
	gsl_rng_set(prand, 0);

	//Pointers for each value in a convolutional layer
	gsl_matrix *vals[laycnt];
	gsl_matrix *zl[laycnt];
	gsl_matrix *da[laycnt];
	gsl_matrix *error[laycnt];

	//Each layer will have its own weight matrix no matter what,
	//but if the layer is convolutional then the indices represent
	//where in the receptive field the weight connects to instead of the neuron in current layer
	//and neuron in previous layer. This means we will still have a weight matrix for every layer.
	gsl_matrix *ws[laycnt - 1];

	//We will still use a vector for biases, but if the layer is convolutional the vector will be
	//1-dimensional.
	gsl_vector *bias[laycnt - 1];

	//Iterate over each layer, initialize each
	for (i = 0; i < laycnt; i++) {
		//Get init settings for this layer
		thislay = (init -> layer_data) + i;

		//Get the previous layer for later use
		if (i > 0) {
			lastlay = (init -> layer_data) + i - 1;
		}
	
		//Get the type, size, and depth of the layer
		thistype = thislay -> type;

		tsize1 = thislay -> size1;

		tsize2 = thislay -> size2;

		tdepth = thislay -> depth;

		tfield = thislay -> rfield;

		//Store type and rfield
		(lays + i) -> type = thistype;
		(lays + i) -> rfield = tfield;

		//Declare doubles for randomization
		double sigma, thisrand;

		//IF the layer is convolutional, it only has one weight matrix and bias.
		//IF the layer is a pooling layer, it isn't calculated with weights & biases, so it has none.
		//IF the layer is traditional, each layer has its own bias and weight matrix.
		switch(thistype) {
			//Traditional network
			case 't':
				//Initialize vectors as 1-dimensional matrices.
				vals[i] = gsl_matrix_calloc(tsize1, 1);
				zl[i] = gsl_matrix_calloc(tsize1, 1);
				da[i] = gsl_matrix_alloc(tsize1, 1);
				error[i] = gsl_matrix_alloc(tsize1, 1);

				(lays + i) -> vals = *(vals[i]);
				(lays + i) -> zl = *(zl[i]);
				(lays + i) -> da = *(da[i]);
				(lays + i) -> error = *(error[i]);

				//For traditional, allocate weight matrix based on size1 of this layer and size1 of last layer.
				if (i > 0) {
					//Get distribution for randomization
					sigma = 1 / sqrt(tsize1);

					//Dimensions of the matrix are based off of number of neurons in this layer and last.
					ws[i] = gsl_matrix_alloc(tsize1, lastlay -> size1);
					bias[i] = gsl_vector_alloc(tsize1);

					//Assign the matrix to the layer
					(lays + i) -> ws = (ws[i]);
					(lays + i) -> bias = (bias[i]);
					
					//Set the weights, iterate over each row
					for (j = 0; j < tsize1; j++) {
						//Iterate over each row entry
						for (k = 0; k < lastlay -> size1; k++) {
							//Get the random number
							thisrand = gsl_ran_gaussian(prand, sigma);

							//Set the weight
							gsl_matrix_set(ws[i], j, k, thisrand);
						}
						//Before finishing up, set the bias
						thisrand = gsl_ran_gaussian(prand, sigma);
						gsl_vector_set(bias[i], j, thisrand);
					}
				}
				break;
			//For a pooling layer, no bias used. Init the weights based on
			//receptive field. Keep weight and bias empty.
			case 'p':
				//Initialize activation vals & related
				vals[i] = gsl_matrix_calloc(tsize1, tsize2);
				zl[i] = gsl_matrix_calloc(tsize1, tsize2);
				da[i] = gsl_matrix_calloc(tsize1, tsize2);
				error[i] = gsl_matrix_calloc(tsize1, tsize2);

				(lays + i) -> vals = *(vals[i]);
				(lays + i) -> zl = *(zl[i]);
				(lays + i) -> da = *(da[i]);
				(lays + i) -> error = *(error[i]);

				if (i > 0) {
					//Dimensions of the matrix are based off of local receptive field
					ws[i] = gsl_matrix_calloc(tfield, tfield);

					//Only one bias for the whole layer
					bias[i] = gsl_vector_calloc(1);

					//Assign the matrix & bias to the layer
					(lays + i) -> ws = (ws[i]);
					(lays + i) -> bias = (bias[i]);
				}
				break;

			//Convolutional layer
			case 'c':
				//Initialize activation vals & related
				vals[i] = gsl_matrix_calloc(tsize1, tsize2);
				zl[i] = gsl_matrix_calloc(tsize1, tsize2);
				da[i] = gsl_matrix_calloc(tsize1, tsize2);
				error[i] = gsl_matrix_calloc(tsize1, tsize2);

				(lays + i) -> vals = *(vals[i]);
				(lays + i) -> zl = *(zl[i]);
				(lays + i) -> da = *(da[i]);
				(lays + i) -> error = *(error[i]);

				if (i > 0) {
					//Get distribution for randomization
					sigma = 1 / sqrt(tsize1);

					//Dimensions of the matrix are based off of local receptive field
					ws[i] = gsl_matrix_alloc(tfield, tfield);

					//Only one bias for the whole layer
					bias[i] = gsl_vector_alloc(1);

					//Assign the matrix & bias to the layer
					(lays + i) -> ws = (ws[i]);
					(lays + i) -> bias = (bias[i]);

					//Set the weights
					for (j = 0; j < tfield; j++) {
						for (k = 0; k < tfield; k++) {
							//Get random number
							thisrand = gsl_ran_gaussian(prand, sigma);

							//Set the matrix
							gsl_matrix_set(ws[i], j, k, thisrand);	
						}
					}
					//Set the bias
					thisrand = gsl_ran_gaussian(prand, sigma);
					gsl_vector_set(bias[i], 0, thisrand);
				}
				
				break;
		}
	}
	//Free the RNG
	gsl_rng_free(prand);

	return lays;
}
