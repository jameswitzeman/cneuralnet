//Header file with some structs for layers and neurons
#include <stdlib.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_matrix.h>
#ifndef _STRUCTS_

#define _STRUCTS_

//Define constants for debugging
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

//This struct holds the number of neurons in each layer, and the number of layers in the net. 
typedef struct netinit_t {
	//Array of integers, these give the neuron count
	//per layer
	int *ncnt;

	//Number of layers
	int layers;
} netinit;

//This struct holds the activation, bias, and weight data for a single layer.
typedef struct layer_t {

	//Vals holds the activation value vector for a layer
	gsl_vector vals;

	//zl holds the z value vector for a layer
	gsl_vector zl;

	//da holds the derivative of the activation vector 
	gsl_vector da;

	//error holds the error vector for the layer
	gsl_vector error;

	//Biases for the layer
	gsl_vector bias;

	//Weight matrix for the layer
	gsl_matrix ws;
} layer;

//This struct holds the necessary data for a convolutional layer. 
//It can also be used to represent a traditional layer (if you dont mind 1D matrices)
typedef struct conv_layer_t {
	//Type character for quick identification of layer type.
	char type;

	gsl_matrix vals;

	gsl_matrix zl;

	gsl_matrix da;

	gsl_matrix error;

	//Because the weight and bias are shared for every neuron, 
	//we should make the weights and biases point to that shared
	//vector and matrix.
	gsl_vector *bias;

	gsl_matrix *ws;

	//We also need the size of the local receptive field, which is the size of the weight array.
	int rfield;
} conv_layer;

//This struct stores the initialization values for a layer of a convolutional network.
typedef struct conv_layer_init_t {
	/* 
	 * Type for the layer. There are three types:
	 * 	t: Traditional network layer, with 1-dimension & fully-connected neurons.
	 * 	c: Convolutional layer, with 2-dimensions plus a third for different "features."
	 * 	p: Pooling layer, same structure as convolutional layer but with different activation algorithm.
	*/
	char type;

	//Size of the layer. Two sizes, in case of a 2-d layer. If the layer is traditional, then size2 doesnt matter.
	int size1;

	int size2;

	//If the layer has 2 dimensions, then we keep a third int for storing the 3rd dimension for a layer.
	int depth;

	//Local receptive field size for each layer
	int rfield;
} conv_layer_init;

//This struct holds all the data needed to initialize a convolutional neural network.
typedef struct conv_init_t {
	//Number of layers in the network
	int layers;

	//Array of initialization data for every layer.
	conv_layer_init *layer_data;
} conv_init;

//This struct holds the gradients of the cost function
typedef struct gradient_t {
	//Weights partials
	gsl_matrix *wgrad;

	//Biases partials
	gsl_vector *bgrad;
} gradient;

#endif
