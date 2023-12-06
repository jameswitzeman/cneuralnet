//This contains prototypes for the netex functions.
//These functions are used for the heavy-duty calculations
//for the training of a non-convolutional neural network.
#ifndef _NETEX_

#define _NETEX_

#include <gsl/gsl_math.h>
#include "structs.h"

/* vectorize_alloc returns a pointer to a vectorized version of the input matrix.
 * It takes each row of the matrix and enters it into the vector.
 * Inputs:
 * 	gsl_matrix *matx
 * Returns
 * 	gsl_vector *out
 */
gsl_vector *vectorize_alloc(gsl_matrix *matx);

/* expected_out takes a label and the size of the output layer,
 * and returns a vector that holds what the output should be.
 * Assumes that the output should only have one neuron set high.
 * Inputs:
 * 	int label: the number that shows which corresponding neuron should be active.
 * 	int size: the size of the output layer.
 * Returns nothing
 */
gsl_vector *expected_out(int label, int size);

/* linear_map is a small function that takes an input with a known,
 * finite domain and maps it linearly to a new domain.
 * Inputs:
 * 	input x;
 * 	old domain minimum m;
 * 	old domain maximum M;
 * 	new domain minimum n;
 * 	new domain maximum N;
 * Returns the mapped value of x as a double.
 */
double linear_map(double x, double m, double M, double n, double N);

/* fsig is a simple implementation of the sigmoid function, commonly
 * used for converting or "squishing" the activations of neurons to
 * a number between 0 - 1.
 * Inputs:
 * 	input x
 * Returns a double, the output of the sigmoid function.
 */
double fsig(double x);

/* fsigv is a "vectorization" of the sigmoid function. It will
 * replace the values in a vector with the values after being sent through
 * the sigmoid function.
 * Inputs:
 * 	input vector pointer vect
 * 	output vector pointer vout
 * Returns void
 */
void fsigv(gsl_vector *vect, gsl_vector *vout);

/* The next two functions are exactly the same, but they compute the
 * derivative of the sigmoid function.
 */
double dsig(double x);

void dsigv(gsl_vector *vect, gsl_vector *vout);

/* vln is the vectorization of the standard log() function.
 * It works like the last two vectoriized math functions.
 */
void vln(gsl_vector *vect, gsl_vector *vout);

/* build_architecture takes in initialization settings and builds a non-convolutional
 * neural network based on these.
 * Inputs: special structure netinit_t which holds the neuron count per layer
 * 	       and layer count.
 * Returns:
 * 	A pointer to an array of network layers set up as requested.
 */
layer *build_architecture(netinit init_vals);

/* scost gets the cost of the output layer. This is actually not necessary for 
 * the training of the network because the backpropagation algorithm only uses
 * the derivative of the cost, but it is useful to provide the user with the
 * actual cost of the output.
 * Inputs:
 * 	gsl_vector *pvals: Vector that holds the expected output
 * 	gsl_vector *aL: Vector that holds the actual output
 * Returns a double value, the cost.
 */
double scost(gsl_vector *pvals, gsl_vector *aL);

/* get_al calculates the activations of a layer based on the activations of the previous layer,
 * as well as the weights and biases for each neuron.
 * Inputs:
 * 	layer *layl: array of network layers (initialized by build_architecture)
 * Returns nothing. Writes directly to the network layer.
 */
void get_al(layer *layl);

/* get_da gets the derivative of the activations of a layer. NOTE: This function assumes
 * that activations are calulated using the SIGMOID function. Other functions
 * should be used for getting da if no portion of your network uses the sigmoid function.
 * Inputs: 
 * 	layer *layl: Layer array initialized by build_architecture
 * Returns nothing. Writes directly to network layer.
 */
void get_da(layer *layl);

/* softmax_a computes the activations of a layer based on the softmax function, NOT the sigmoid
 * function. This is intended to be used only for the output layer, but I guess
 * you could use it on any layer.
 * Inputs:
 * 	layer *layl; layer array initialized by build_architecture
 * Returns nothing. Writes directly to network layer.
 */
void softmax_a(layer *layl);

/* get_lerr calculates the error in the output layer of the network. This function assumes
 * that the cost function is log-likelihood or cross-entropy (derivative for both is the same).
 * Inputs:
 * 	layer *layl: layer array initialized by build_architecture
 * 	gsl_vector *correct_vals: a vector that holds the expected output
 * 							  for the image it was given.
 * Returns nothing. Does not affect the correct_vals vector, writes to network.
 */
void get_lerr(layer *layl, gsl_vector *correct_vals);

/* get_err uses backpropagation to calculate the error of a layer based on the 
 * error of the layer ahead of it. It assumes that the current layer's activations
 * are calculated with the sigmoid function.
 * Inputs: 
 * 	layer *layl
 * Returns nothing, writes to network.
 */
void get_err(layer *layl);

/* get_grad calculates the gradient of a layer given the error vector and activation
 * vector of the previous layer.
 * Inputs:
 * 	layer *layl
 * 	gradient *grad: gradient struct for holding gradients. See structs.h
 * Returns nothing
 */
void get_grad(layer *layl, gradient *grad);


/* gradient_descend takes a network and a calculated gradient, and adjusts the weights
 * and biases by a small step dn.
 * Inputs:
 * 	netinit net_init: Basic info on the network
 * 	layer *lays: the network architecture
 * 	gradient *grad: the pre-computed gradient of the network
 * 	double dn: The step size for descent. 
 * 		   Note that this step size should already be divided by examples per block.
 * 	double regco: The regularization coefficient (NOT PARAMETER!!)
 * Returns nothing
 */
void gradient_descend(netinit net_init, layer *lays, gradient *grad, double dn, double regco);

/* backprop takes all these functions and calculates the gradient of the network -- as well
 * as the cost -- for a single example. It should write to a pre-allocated gradient struct
 * and return the cost as a double. Note that the function assumes the input data has already
 * been written (because who knows what the data is).
 *
 * Inputs:
 * 	int debug: Debug integer for toggling what debug info is printed to console
 * 	netinit net_init: struct that just holds important data about the network like
 * 			  neuron count per layer and layer count.
 * 	layer *lays: Neural network struct
 *	gradient *grad: Gradient struct to write to
 *	gsl_vector *corrects: Vector that holds the expected output
 * Returns: double cost: the actual cost for this example.
 */
double backprop(int debug, netinit net_init, layer *lays, gradient *grad, gsl_vector *corrects);

/* test takes a network which has already had an input written to it, and tests the accuracy of
 * the output. If the network correctly guessed the input, it returns a 1. If not, it returns a 0.
 *
 * Inputs:
 * 	int debug: Debug integer for toggling what debug info is printed to console
 * 	netinit net_init: struct that contains info on the network
 * 	layer *lays: Neural Network struct
 * 	gsl_vector *corrects: expected output vector.
 * Returns:
 * 	integer that should only be 0 or 1.
 */
int test(int debug, netinit net_init, layer *lays, gsl_vector *corrects);

#endif
