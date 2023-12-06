/* This header contains prototypes for initializing and training a convolutional neural network.
 * It depends on the other init & train header netex.h for a few things.
 */
#ifndef _CONV_

#define _CONV_

#include <gsl/gsl_math.h>
#include "structs.h"
#include "netex.h"

/* 
 * conv_architecture takes in an initialization structure and uses it to initialize 
 * a convolutional network.
 * Inputs:
 * 	conv_init *init: The initialization settings for the network
 * Returns an array of convolutional layers.
 */
conv_layer *conv_architecture(conv_init *init);

#endif
