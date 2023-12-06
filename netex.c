/* This script utilises the GNU Scientific Library
 * to generate a neural network arcitecture,
 * and train it using the backpropagation algorithm.
 * To do this efficiently, a hefty amount of vector and
 * matrix math is used.
 */

#include <stdio.h>
#include <stdlib.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#include <gsl/gsl_cdf.h>
#include <gsl/gsl_math.h>
#include <math.h>
#include "structs.h"
#include "netex.h"

//Remember that this allocates a vector that should be freed.
gsl_vector *vectorize_alloc(gsl_matrix *matx) {
	int msize1 = matx -> size1;
	int msize2 = matx -> size2;
	int vsize = msize1 * msize2;

	//Allocate space for the vector
	gsl_vector *vout = gsl_vector_alloc(vsize);

	//Placeholder double
	double thisval;

	for (int i = 0; i < msize1; i++) {
		for (int j = 0; j < msize2; j++) {
			thisval = gsl_matrix_get(matx, i, j);
			//Make sure to shift the index by the size of one row each time
			//we start loading in a new row
			gsl_vector_set(vout, j + (i * msize1), thisval);
		}
	}
}

//This function gets the proper output layer values given a label.
//It assumes that only one neuron should be active in the output, and the label
//corresponds to that neuron. Intended for use in image recognition.
//NOTE: The vector it creates MUST BE FREED! Not automatic
gsl_vector *expected_out(int label, int size) {
	//Allocate the vector
	gsl_vector *vect = gsl_vector_alloc(size);

	//Set each vector value
	int l;
	for (l = 0; l < size; l++) {
		if (l == label) {
			//If we're at the label, set that vector val to 1
			gsl_vector_set(vect, label, 1);
		}
		//Else, 0
		else {
			gsl_vector_set(vect, l, 0);
		}
	}
	return vect;
}

//This function is a small, simple linear map which takes
//the maximum and minimum of a set of values and one value in that set.
//It also takes a new min and max, and maps the value to that.
double linear_map(double x, double m, double M, double n, double N) {
	double out = ((N - n) / (M - m)) * (x - m) + n;
	return out;
}

//Sigmoid function
double fsig(double x) {
	double out = 1/(1 + exp(-1 * x));
	return out;
}

//Sig func for a vector. Outputs result to out vector
void fsigv(gsl_vector *vect, gsl_vector *vout) {
	double gotvect;
	for (int i = 0; i < vect -> size; i++) {
		gotvect = gsl_vector_get(vect, i);
		gotvect = fsig(gotvect);
		gsl_vector_set(vout, i, gotvect);
	}
}

// Derivative of the sigmoid function
double dsig(double x) {
	double out = fsig(x) * (1 - fsig(x));
	if (out != out) {
		out = 0;
	}
	return out;
}

// dsig for a vector.
void dsigv(gsl_vector *vect, gsl_vector *vout) {
	double gotvect;
	for (int i = 0; i < vect -> size; i++) {
		gotvect = gsl_vector_get(vect, i);
		gotvect = dsig(gotvect);
		gsl_vector_set(vout, i, gotvect);
	}
}

//Natural logarithm for a vector
void vln(gsl_vector *vect, gsl_vector *vout) {
	double gotvect;
	for (int i = 0; i < vect -> size; i++) {
		gotvect = gsl_vector_get(vect, i);
		gotvect = log(gotvect);
		gsl_vector_set(vout, i, gotvect);
	}
}

//This function takes in a number of layers and neurons for each layer
//and builds the math objects necessary for calculations
layer *build_architecture(netinit init_vals) {
	//Some integers that will be useful
	int i, j, k;

	//Init a pointer to the layer array that we will
	//return at the end
	layer *lays = malloc(init_vals.layers * sizeof(layer));

	//Initialize the GSL random number generator. First, we allocate space for it.
	gsl_rng *prand = gsl_rng_alloc(gsl_rng_mt19937);

	//Seed the RNG
	gsl_rng_set(prand, 0);

	//Now we have to allocate a weight matrix, activation vector, and bias vector for every
	//layer. First, we make pointers which will point to an array of each.
	gsl_vector *activations[init_vals.layers];
	gsl_vector *das[init_vals.layers];
	gsl_vector *zls[init_vals.layers];
	gsl_vector *biases[init_vals.layers - 1];
	gsl_matrix *weights[init_vals.layers - 1];
	gsl_vector *errors[init_vals.layers];

	//For each layer, create an activation & bias vector with the proper size
	for (i = 0; i < init_vals.layers; i++) {
		activations[i] = gsl_vector_calloc(*(init_vals.ncnt + i));
		biases[i] = gsl_vector_alloc(*(init_vals.ncnt + i));
		das[i] = gsl_vector_alloc(*(init_vals.ncnt + i));
		zls[i] = gsl_vector_alloc(*(init_vals.ncnt + i));
		errors[i] = gsl_vector_alloc(*(init_vals.ncnt +i));

		//Assign the activation, bias, zl, and error vectors to the corresponding layer
		(lays + i) -> vals = *(activations[i]);
		(lays + i) -> zl = *(zls[i]);
		(lays + i) -> da = *(das[i]);
		(lays + i) -> bias = *(biases[i]);
		(lays + i) -> error = *(errors[i]);

		//Make a weight matrix, but only if we're not on the starting
		//layer.
		double sigma, thisrand;
		
		if (i > 0) {
			//Integer for storing the number of weights each neuron has
			int wsize = (lays + i) -> vals.size;
			weights[i] = gsl_matrix_alloc(*(init_vals.ncnt + i), *(init_vals.ncnt +i - 1));

			//Assign the weight matrix to the corresponding layer.
			(lays + i) -> ws = *(weights[i]);
			
			//Set the weight for each layer (except the first)
			//Iterate for each row
			for (j = 0; j < (weights[i] -> size1); j++) {

				//Iterate for each row entry
				for (k = 0; k < (weights[i] -> size2); k++) {
					//Randomly initialize each weight as a selection
					//from a gaussian distribution with standard deviation
					//1/sqrt(nin)
					sigma = 1 / sqrt(wsize);
					thisrand = gsl_ran_gaussian(prand, sigma);
					gsl_matrix_set(weights[i], j, k, thisrand);
				}
			}
		}

		//For each neuron in this layer, set biases.
		
		for (j = 0; j < (activations[i] -> size); j++) {
			sigma = 1 / sqrt(activations[i] -> size);
			gsl_vector_set(biases[i], j, gsl_ran_gaussian(prand, sigma));
		}
		
		
	}

	weights[0] = gsl_matrix_calloc(*(init_vals.ncnt), 1);

	//Free the memory allocated for RNG
	gsl_rng_free(prand);

	return lays;
}

//Get single example cost, the log-likelihood cost function.
//Should only be used on the output layer.
double scost(gsl_vector *pvals, gsl_vector *aL) {
	int label, i, psize, asize;

	psize = pvals -> size;
	asize = aL -> size;

	//output variable
	double out;

	//This vector holds output activations.
	gsl_vector *templ = gsl_vector_alloc(aL -> size);
	gsl_vector_memcpy(templ, aL);

	//Get the label
	for (i = 0; i < psize; i++) {
		if (gsl_vector_get(pvals, i) == 1) {
			label = i;
		}
	}

	out = gsl_vector_get(aL, label);
	out = log(out);
	out *= -1;

	gsl_vector_free(templ);
	return out;
}

//This function is the first step of backpropagation. It computes the activations and zl values
//for each neuron of a single layer. It should return nothing and directly insert its calculations.
void get_al(layer * layl) {
	//Iteration ints
	int j; 
	
	//Allocate a temporary matrix for math, plus some convenience pointers
	gsl_matrix *tempm = gsl_matrix_alloc(layl -> ws.size1, layl -> ws.size2);
	gsl_vector *row = gsl_vector_alloc(layl -> ws.size2);
	gsl_vector *al1 = &((layl - 1) -> vals);
	gsl_vector *bl = &(layl -> bias);
	gsl_vector *out = &(layl -> vals);

	//Multiply each column by the vector a(l-1)
	gsl_matrix_memcpy(tempm, &(layl -> ws));
	gsl_matrix_scale_columns(tempm, al1);

	//Go row by row, put the row in a vector.
	for (j = 0; j < tempm -> size1; j++) {
		//Get a row from the matrix
		gsl_matrix_get_row(row, tempm, j);

		//Sum the row, put it into the out vector
		gsl_vector_set(out, j, gsl_vector_sum(row));

	}

	//Sum the out vector with the bias vector
	gsl_vector_add(out, bl);

	//This is zl. Copy it into zl.
	gsl_vector_memcpy(&(layl -> zl), out);

	//finally, take the sigmoid of the out vector.
	fsigv(out, out);

	gsl_matrix_free(tempm);
	gsl_vector_free(row);
}

//This function calculates the derivative of the activations for a layer.
//This is automatically done when the cost is calculated for the output layer,
//but it will need to be done for the other layers.
void get_da(layer *layl) {
	//Get zl
	gsl_vector *thisz = &(layl -> zl);
	//Get output vector da
	gsl_vector *thisda = &(layl -> da);

	//Take dsig of zl, put it in da.
	dsigv(thisz, thisda);
}

//This is the softmax function, for computing activations in the output layer.
//It's assumed that the previous activation function has already been performed,
//and the Z-values have been calculated.
void softmax_a(layer *layl) {
	gsl_vector *z = &(layl -> zl);
	double entry, sum;
	int i;

	//Get the denominator of the softmax; the sum of all output activations
	for (i = 0; i < z -> size; i++) {
		entry = gsl_vector_get(z, i);
		entry = exp(entry);
		sum += entry;
	}

	for (i = 0; i < z -> size; i++) {
		entry = gsl_vector_get(z, i);
		entry = exp(entry);
		entry = entry / sum;
		gsl_vector_set(&(layl -> vals), i, entry);
	}

}

//This function computes the error of the output layer. It takes the correct values
//as an input.
void get_lerr(layer *layl, gsl_vector *correct_vals) {
	//We need two vectors; da and a vector that holds the difference between
	//the output layer and the correct values. This difference vector does not
	//exist, so we allocate space for a temporary one.
	
	//Temp vector for holding the errors before save	
	gsl_vector *dc = gsl_vector_alloc(layl -> vals.size);

	//Copy value into dc
	gsl_vector_memcpy(dc, &(layl -> vals));

	//Take the difference
	gsl_vector_sub(dc, correct_vals);
	
	//That's it! save the error.
	gsl_vector_memcpy(&(layl -> error), dc);

	//Free temporary vector
	gsl_vector_free(dc);
}

//This function computes the error of a layer in terms of the layer ahead of it.
//It takes as input a layer array.
void get_err(layer *layl) {
	int i;
	double sum;
	gsl_vector *mrow;

	//Initialize temporary matrix and vector for operations
	gsl_matrix *mflop = gsl_matrix_alloc(((layl+1) -> ws.size2), ((layl+1) -> ws.size1));
	mrow = gsl_vector_alloc(mflop -> size2);

	//Copy weight matrix to temp matrix, transpose in the process
	gsl_matrix_transpose_memcpy(mflop, &((layl + 1)->ws));

	//Get the error of the layer ahead
	gsl_vector *err = &((layl + 1) -> error);

	//Multiply error and the matrix
	gsl_matrix_scale_columns(mflop, err);

	//Now we convert this matrix into a vector by summing every row.
	//Iterate over each row
	for (i = 0; i < mflop -> size1; i++) {
		//Get the row
		gsl_matrix_get_row(mrow, mflop, i);

		//Sum the row
		sum = gsl_vector_sum(mrow);

		//Put the sum into the error vector
		gsl_vector_set(&(layl -> error), i, sum);

		//Multiply this sum by da
		gsl_vector_mul(&(layl -> error), &(layl -> da));


	}

	//Free temporary matrix and vector
	gsl_vector_free(mrow);
	gsl_matrix_free(mflop);
	
}

//This function gets the gradient of a layer given the error vector and
//activation vector in the last layer.
void get_grad(layer *layl, gradient *grad) {
	int j, k;
	double entry;

	//Get the bias gradient first.
	gsl_vector *pbgrad = &(layl -> error);
	//Get the activation of the last layer.
	gsl_vector *all = &((layl - 1) -> vals);

	//Put the error into the bias grad
	gsl_vector_memcpy(grad -> bgrad, pbgrad);

	//Do the matrix multiplication
	for (j = 0; j < grad -> wgrad -> size1; j++) {
		for (k = 0; k < grad-> wgrad -> size2; k++) {
			entry = gsl_vector_get(all, k);
			entry *= gsl_vector_get(pbgrad, j);
			gsl_matrix_set(grad -> wgrad, j, k, entry);
		}
	}
	
}

void gradient_descend(netinit net_init, layer *lays, gradient *grad, double dn, double regco) {
	//Declare variables
		//Iteration ints
	int i;
	
		//size of output layer, number of layers, index of output layer
	int sout, laynum, outindex; 

		//Gradient & layer vectors for iteration
	gradient *this_grad;
	layer *this_lay;

		//Get the output layer size & layer count
	laynum = net_init.layers;
	outindex = laynum - 1;
	sout = net_init.ncnt[laynum - 1];

	//Iterate over each layer (except input)
	for (i = 1; i < laynum; i++) {
		//Get current gradient
		this_grad = (grad + i);

		//Get current layer
		this_lay = (lays + i);

		//Adjust weights
			//Scale gradient by dn
		gsl_matrix_scale(this_grad -> wgrad, dn);
			//Scale weights by regco
		gsl_matrix_scale(&(this_lay -> ws), regco);
			//Subtract weight gradient from weights
		gsl_matrix_sub(&(this_lay -> ws), this_grad -> wgrad);

		//Adjust biases
			//Scale gradient by dn
		gsl_vector_scale(this_grad -> bgrad, dn);
			//Subtract biases gradient from biases
		gsl_vector_sub(&(this_lay -> bias), this_grad -> bgrad);
	}
	
}

//Training function. See netex.h for details.
double backprop(int debug, netinit net_init, layer *lays, gradient *grad, gsl_vector *corrects) {
	//Declare variables
		//Iteration ints
	int i, j, k;
	
		//size of output layer, number of layers, index of output layer
	int sout, laynum, outindex; 
	double cost;

		//Get the output layer size & layer count
	laynum = net_init.layers;
	outindex = laynum - 1;
	sout = net_init.ncnt[laynum - 1];

	gsl_vector *vout = &((lays + outindex) -> vals);

	//Feedforward. Get the activations for every layer (besides input)
		//Get the activations in hidden layers first (only iterate over them)
		//Also get derivatives of activations.
	for (i = 1; i < laynum; i ++) {
		get_al(lays + i);
		get_da(lays + i);
	}
		//Get activations in the output layer using softmax.
	softmax_a(lays + outindex);
	
	//Calculate and save the cost for this example.
	cost = scost(corrects, vout);
	
	//Get error in the output layer.
	get_lerr((lays + outindex), corrects);
	
	//Backpropagate the error.
	for (i = outindex - 1; i > 0; i--) {
		get_err(lays + i);
	}
	
	//Compute and save the gradient for this example.
	for (i = 1; i < laynum; i++) {
		get_grad(lays + i, grad + i);
	}

	//Debug: Print output layer activations
	if (debug & BIT3) {
		for (j = 0; j < sout; j++) {
			printf("    Neuron %d: ", j);
			printf("[ %.3lf ]\n", gsl_vector_get(&((lays + outindex) -> vals), j));
		}
	}

	//Debug: Printout the output layer gradient.
	if (debug & BIT4) {
		//Iterate over each row in the matrix
		for (j = 0; j < sout; j++) {
			printf("Neuron %d: [", j);
			//Iterate over each entry in a matrix row
			for (k = 0; k < (grad + outindex) -> wgrad -> size2; k++) {
				printf(" %.3lf ", gsl_matrix_get(((grad + outindex) -> wgrad), j, k));
			}
			printf("] [ %.3lf ]\n", gsl_vector_get(((grad + outindex) -> bgrad), j));
		}
	}
	
	//Return the cost.
	return cost;
}

//Testing function
int test(int debug, netinit net_init, layer *lays, gsl_vector *corrects) {
	//Declare variables
		//Iteration ints
	int i;
	
		//size of output layer, number of layers, index of output layer, 
		//the network's guess, the correct answer
	int sout, laynum, outindex, maxindex, label; 
	
		//Get the output layer size & layer count
	laynum = net_init.layers;
	outindex = laynum - 1;
	sout = net_init.ncnt[laynum - 1];

		//Shortcut for output vector
	gsl_vector *vout = &((lays + outindex) -> vals);

	//Feedforward. Get the activations for every layer (besides input)
		//Get the activations in hidden layers first (only iterate over them)
		//Also get derivatives of activations.
	for (i = 1; i < laynum; i ++) {
		get_al(lays + i);
		get_da(lays + i);
	}
	softmax_a(lays + outindex);

	//Get the label
	label = gsl_vector_max_index(corrects);

	//Check the output layer, find the maximum BUT ONLY if max > .5
	if (gsl_vector_max(vout) > 0.25) {
		maxindex = gsl_vector_max_index(vout);
	}
	//If max isnt large enough, then set maxindex to impossible value
	else {
		maxindex = -1;
	}

	//The network's guess is correct if the max index is equal to the label.
	//Return 1 to signify a correct guess.
	if (label == maxindex) {
		return 1;
	}

	//Else, its WRONG!
	return 0;

}
