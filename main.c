#include <stdio.h>
#include <stdlib.h>
#include <gsl/gsl_randist.h>
#include <math.h>
#include <sys/time.h>
#include <time.h>
#include <gsl/gsl_cdf.h>
#include "netex.h"
#include "files.h"
/*
 *This script should utilize the structs and functions provided and 
 *use them with the MNIST database (in .csv format) to train a simple
 *neural net to recognize handwritten digits.
 */

#define BIT0 0x01
#define BIT1 0x02
#define BIT2 0x04
#define BIT3 0x08
#define BIT4 0x10
#define BIT5 0x20
#define BIT6 0x40
#define BIT7 0x80
#define BIT8 0x100
#define BIT9 0x200
#define BIT10 0x400

//The layers desired for the net
#define LAYERS 3

//The number of examples to train on per block.
//The average cost will be calculated based on this.
#define EX_BLOCK 1

//The number of blocks to use. Gradient descent is not applied
//until a block 5as been processed.
#define BLOCK_NUM 50000

#define EPOCHS 50

//This constant represents the small step to nudge each weight and bias
//in the direction of the gradient.
#define DN 0.025

//Regularization parameter. Here, we skip all the fun derivatives and 
//use L2 regularization by adding a coefficient to w during gradient descent
#define LAMBDA 2

int train(int debug, int nnum[LAYERS], layer *lays, gradient *grad, gradient *avg_grad, hash_table *plookup, hash_entry *pvals);

double test(int debug, layer *lays, hash_table *plookup, hash_entry *pvals, int tstcnt);

int main() {

	//Byte for indicating debug output.
	//BIT0 = Get Image Debug
	//BIT1 = Get Proper Vector Debug
	//BIT2 = Feedforward debug
	//BIT3 = Error in activation debug
	//BIT4 = Gradient debug
	//BIT5 = Check the net output after a block
	//BIT6 = Step size debug
	//BIT7 = Load the pre-trained settings file
	//BIT8 = Set this to enable training. If not set, only testing
	//BIT9 = Don't write the settings to file
	//BIT10 = Printout cost every epoch
	int debug = BIT7 | BIT8 | BIT10;

	int g, i, j, k, trained = 1;

	double score, time;

	//Define layer array
	int nnum[LAYERS] = {784, 30, 10};
	netinit init_vals;
	init_vals.ncnt = &nnum[0];
	init_vals.layers = LAYERS;

	//This layer struct array holds the activations, biases, and weights for
	//each layer
	layer *lays;

	//Initialize structs for the gradient and average gradient.
	gradient *grad, *avg_grad;
	grad = malloc(LAYERS * sizeof(gradient));
	avg_grad = malloc(LAYERS * sizeof(gradient));
	for (i = 1; i < LAYERS; i++) {
		(grad + i) -> wgrad = gsl_matrix_calloc(nnum[i], nnum[i - 1]);
		(grad + i) -> bgrad = gsl_vector_calloc(nnum[i]);
		(avg_grad + i) -> wgrad = gsl_matrix_calloc(nnum[i], nnum[i - 1]);
		(avg_grad + i) -> bgrad = gsl_vector_calloc(nnum[i]);
	}


	//Build the arcitecture
	lays = build_architecture(init_vals);

	//If desired, load in pre-trained settings
	if (debug & BIT7) {
		printf("Getting settings from file...\n");
		FILE *netf = fopen("net.txt", "r");

		read_net(netf, lays);

		fclose(netf);
	}
	
	//Get the right results for each digit
	struct htable_t *plookup; 

	//This struct will point to the ideal output layer
	struct hentry_t *pvals;

	plookup = get_right_digits();

	//Train the net over a set of epochs
	//Test each time, print the accuracy.
	

	
	for (g = 0; g < EPOCHS; g++) { 
		printf("Epoch %d\n", g);

		//Get the time elapsed per epoch
		clock_t start, stop;
		start = clock();

		//Only train if bit8 is not set
		if (debug & BIT8) {
			trained = train(debug, nnum, lays, grad, avg_grad, plookup, pvals);
		}

		score = test(debug, lays, plookup, pvals, 9999);
		printf("  Score: %lf\n", score);

		stop = clock();
		time = (double)(stop - start) / CLOCKS_PER_SEC;

		printf("  Time: %.3lfsec\n", time);

		//Save the network settings to a file!
		if (!trained && !(debug & BIT9)) {
			printf("Saving settings...\n");
			FILE *outf = fopen("net.txt", "w+");
		
			//Iterate over each layer except input layer.
			for (int i = 1; i < LAYERS; i++) {
				//Iterate over each neuron in this layer.
				for (j = 0; j < nnum[i]; j++) {
					//Iterate over each neuron in the last layer
					for (k = 0; k < nnum[i - 1]; k++) {
						//Print the weight k in row j
						fprintf(outf, "%lf,", gsl_matrix_get(&((lays + i) -> ws), j, k));
					}
					fprintf(outf, "|%lf\n", gsl_vector_get(&((lays + i) -> bias), j));
				}
				fprintf(outf, "L\n", i + 1);
			}
			//fprintf(outf, EOF);
		
			//Close the file
			fclose(outf);
		}
	}
	
	return 0;
}


//FUNCTIONS

//Testing function
double test(int debug, layer *lays, hash_table *plookup, hash_entry *pvals, int tstcnt) {
	int size = lays -> vals.size, label, i, j, k, correct, guess, gotit;
	//Size of output vector
	int sout = (lays + LAYERS - 1) -> vals.size;
	FILE *test = fopen("MNIST_CSV/mnist_test.csv", "r");

	//Shortcut for output layer
	gsl_vector *output = &((lays + LAYERS - 1) -> vals);

	//Get the first image
	double *flout = get_img(test, size + 1);
	//Doubles for the accuracy
	double thisval, nextval, highest, avg;
	j = 0;
	//Keep going until end of file
	correct = 0;
	while (j < tstcnt) {
		j++;
		//Get the label
		label = (int) flout[0];

		//Put the image into the input vector
		gsl_vector *input = &(lays -> vals);

		for (i = 0; i < size; i++) {
			gsl_vector_set(input, i, flout[i + 1]);
			//Debug message. The values should match up.
			if (debug & BIT0) {
				printf(
					"Image value %d: %.3lf  Input Value %d: %.3lf\n",
					i,
					flout[i + 1],
					gsl_vector_get(&((lays) -> vals), i)
				);
			}
		}

		//Feedforward. Get activations and zl values.
		//Iterate through each layer and get al & zl.
		for (i = 1; i < LAYERS; i ++) {
			get_al(lays + i);
		}
		//Debug. Print the zl and al values for output layer.
		if (debug & BIT2) {
			printf("In output layer:\n");
			for (i = 0; i < sout; i++) {
				printf(
					"al %d: %.3lf    zl %d: %.3lf    da %d: %.3lf\n",
					i,
					gsl_vector_get(&((lays + LAYERS - 1) -> vals), i),
					i,
					gsl_vector_get(&((lays + LAYERS - 1) -> zl), i),
					i,
					gsl_vector_get(&((lays + LAYERS - 1) -> da), i)
				);
			}
			printf("\n");
		}

		//Get the correct output
		//Get the expected output y
		pvals = lookup(plookup, label);

		//Allocate memory for putting the expected output into a vector.
		gsl_vector *propers = gsl_vector_alloc(sout);

		//Debug for correct vector
		if (debug & BIT1) {
			printf("For an image of: %d\n", label);
		}

		//Put correct vector into propers
		for (i = 0; i < sout; i++) {
			gsl_vector_set(propers, i, (double) *(pvals -> data + i));

			//Debug. Print out propers vector.
			if (debug & BIT1) {
				printf("Neuron %d should be: %.lf\n", i, gsl_vector_get(propers, i));
			}
		}

		//Printout debug. Shows the output of the net
		if (debug & BIT5) {
			printf("For label: %d [ ", label);
			for (int p = 0; p < sout; p++) {
				printf("%.3lf ", gsl_vector_get(&(lays + LAYERS - 1) -> vals, p));
			}
			printf("]  ");
		}

		//Loop through each entry in the output, find the highest value
		highest = gsl_vector_get(output, 0);
		guess = 0;
		gotit = 0;

		for (k = 0; k < sout - 1; k++) {
			thisval = gsl_vector_get(output, k);
			nextval = gsl_vector_get(output, k + 1);

			if (nextval > highest) {
				highest = nextval;
				guess = k + 1;
			}
		}
		if (debug & BIT5) {
			printf("Guess: %d  ", guess);
		}

		//Make sure we only register it as a guess if the highest value
		//is larger than .15. If it isnt, automatically register it as wrong.
		if (highest <= 0.15) {
			guess = 10;
		}

		//Check if the highest value is correct. If it is, then add to the correct count.
		if (guess == label) {
			correct++;
			if (debug & BIT5) {
				printf("Correct\n");
			}
		}
		else {
			if (debug & BIT5) {
				printf("Incorrect\n");
			}
		}

		gsl_vector_free(propers);

		//Get the next image
		free(flout);
		flout = get_img(test, size + 1);
	}
	avg = ((float) correct)/((float) tstcnt);
	avg *= 100;
	fclose(test);

	return avg;
}

//Training function
int train(int debug, int nnum[LAYERS], layer *lays, gradient *grad, gradient *avg_grad, hash_table *plookup, hash_entry *pvals) {
	//Some ints for iteration, plus a label for storing right answers
	int g, h, i, j, k, label, wsize;

	//This float array holds the image input data
	double *flimg;

	//Doubles for cost of the output and the average cost over the entire set of
	//inputs, as well as a double for adding variance to gradient descent steps
	double imgcost, cavg, stepvar, regco;

	
	//This vector holds the correct output for a training example
	gsl_vector *propers;


	//Set initial average cost to 0
	cavg = 0;

	//Open the file
	FILE *file = fopen("MNIST_CSV/mnist_train.csv", "r");

	//Loop for every block of training examples
	for (g = 0; g < BLOCK_NUM; g++) {
		//Reset the average gradient vector
		for (i = 1; i < LAYERS; i++) {
			gsl_matrix_set_zero((avg_grad + i) -> wgrad);
			gsl_vector_set_zero((avg_grad + i) -> bgrad);
		}
	
		//Initialize random number generation for gradient descent
		gsl_rng *pr = gsl_rng_alloc(gsl_rng_mt19937);
		gsl_rng_set(pr, g);

		double cst = 0;

		//Loop for the set number of examples per block
		for (h = 0; h < EX_BLOCK; h++) {
			//Get an image. Remember, the very first number is the label.
			flimg = get_img(file, nnum[0] + 1);

			//Load the image into input layer.
			//Loop through each pixel and copy it into 
			//activations of layer 0.

			//Get the label of the image
			label = (int) flimg[0];

			//Import the image
			for (i = 1; i < nnum[0] + 1; i++) {
				gsl_vector_set(&((lays) -> vals), i - 1, flimg[i]);

				//Debug message. The values should match up.
				if (debug & BIT0) {
					printf(
						"Image value %d: %.3lf  Input Value %d: %.3lf\n",
						i - 1,
						flimg[i],
						gsl_vector_get(&((lays) -> vals), i - 1)
					);
				}
			}

			//Get the expected output y
			pvals = lookup(plookup, label);

			//Allocate memory for putting the expected output into a vector.
			propers = gsl_vector_alloc(nnum[LAYERS - 1]);

			//Debug for correct vector
			if (debug & BIT1) {
				printf("For an image of: %d\n", label);
			}

			//Put correct vector into propers
			for (i = 0; i < nnum[LAYERS - 1]; i++) {
				gsl_vector_set(propers, i, (double) *(pvals -> data + i));

				//Debug. Print out propers vector.
				if (debug & BIT1) {
					printf("Neuron %d should be: %.lf\n", i, gsl_vector_get(propers, i));
				}
			}
			
			//Feedforward. Get activations and zl values, as well as da.
			//Iterate through each layer and get al & zl.
			for (i = 1; i < LAYERS; i ++) {
				get_al(lays + i);
				get_da(lays + i);
			}
			//Get activation in the output layer
			//softmax_a(lays + LAYERS - 1);

			double sscost = scost(propers, &((lays + LAYERS - 1) -> vals));
			cst += scost(propers, &((lays + LAYERS - 1) -> vals))/EX_BLOCK;

			//Debug. Print the zl and al values for output layer.
			if (debug & BIT2) {
				printf("In output layer:\n");
				for (i = 0; i < nnum[LAYERS - 1]; i++) {
					printf(
						"al %d: %.3lf    zl %d: %.3lf    da %d: %.3lf\n",
						i,
						gsl_vector_get(&((lays + LAYERS - 1) -> vals), i),
						i,
						gsl_vector_get(&((lays + LAYERS - 1) -> zl), i),
						i,
						gsl_vector_get(&((lays + LAYERS - 1) -> da), i)
					);
				}
				printf("\n");
			}

			//Get the error vector for output layer.
			get_lerr((lays + LAYERS - 1), propers);

			//Backpropagate the error. Use output error to find the error
			//for each layer.
			for (i = LAYERS - 2; i > 0; i--) {
				get_err(lays + i);
			}

			//Debug: print output layer error vector, plus the error vect before the output.
			if (debug & BIT3) {
				for (i = 1; i < LAYERS; i++) {
					printf("Error vector in layer %d:\n[ ", i + 1);
					for (j = 0; j < nnum[i]; j++) {
						printf("%lf ", gsl_vector_get(&((lays + i) -> error), j));
					}
					printf("]\n\n");
				}
			}

			//Calculate and save the gradient.
			//the gradient for weights is the product of the last layer's activation
			//and the error of the layer.
			for (i = 1; i < LAYERS; i++) {
				get_grad(lays + i, grad + i);
			}
			
			//Debug. Printout some gradient layers
			if (debug & BIT4) {
				//Use this for loop to choose
				//how many layers to printout
				for (i = LAYERS - 1; i < LAYERS; i++) {
					//For each neuron in the layer
					for (j = 0; j < nnum[i]; j++) {
						printf("[");
						//For each neuron in the last layer
						for (k = 0; k < nnum[i - 1]; k++) {
							printf("%.3lf ", gsl_matrix_get((grad + i) -> wgrad, j, k));
						}
						printf(" [%.3lf ]\n", gsl_vector_get((grad + i) -> bgrad, j));
					}
				}
			}
			//Add this gradient to the average gradient.
			for (i = 1; i < LAYERS; i++) {
				gsl_matrix_add((avg_grad + i) -> wgrad, (grad + i) -> wgrad);
				gsl_vector_add((avg_grad + i) -> bgrad, (grad + i) -> bgrad);
			}
	

			//Free correct answer vector
			gsl_vector_free(propers);
			free(flimg);
		}
		//Once a block has completed, begin gradient descent.
		//Scale the average gradient by n/m,
		//then take the step.
		
		//Instead of same-size steps, vary the step each time.
		//We add a random gaussian-distributed variance.
		stepvar = DN;

		//Free the RNG
		gsl_rng_free(pr);
		
		//Printout the step for debug
		if (debug & BIT6) {
			printf("Step size: %.3lf ", stepvar);
		}

		for (i = 1; i < LAYERS; i++) {
			wsize = ((lays + i) -> ws.size1) * ((lays + i) -> ws.size2);
			//Calculate the regularization coefficient
			regco = 1 - (LAMBDA * (DN /(BLOCK_NUM * EX_BLOCK)));
			//regco = 1;

			gsl_matrix_scale((avg_grad + i) -> wgrad, stepvar / EX_BLOCK);
			gsl_matrix_scale(&(lays + i) -> ws, regco);
			gsl_matrix_sub(&((lays + i) -> ws), (avg_grad + i) -> wgrad);

			gsl_vector_scale((avg_grad + i) -> bgrad, stepvar / EX_BLOCK);
			gsl_vector_sub(&((lays +i) -> bias), (avg_grad + i) -> bgrad);
		}

		if ((debug & BIT5)) {
			//Printout the input label and output.
			printf("Label: %d [", label);
			for (i = 0; i < nnum[LAYERS -1]; i++) {
				printf("%.2lf ", gsl_vector_get(&((lays + LAYERS - 1) -> vals), i));
			}	
			printf("] COST: %lf\n", cst);
		}
		//Get the cost average for the entire training set.
		cavg += cst / BLOCK_NUM;

	}
	if (debug & BIT10) {
		printf("  Cost: %lf  ", cavg);
	}

	fclose(file);
	return 0;
}	

