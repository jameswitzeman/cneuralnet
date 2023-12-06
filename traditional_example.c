/* traditional_example.c is a simple example script utilizing this c-based neural networking
 * library. It uses the MNIST database in .csv format to train a neural network
 * to recognize handwritten digits.
 * Dependencies:
 * 	gnu scientific library
 * 	netex.h
 * 	files.h
 * 	structs.h
 */

//Standard library headers
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <time.h>
#include <string.h>
#include <unistd.h>

//GSL headers
#include <gsl/gsl_math.h>
#include <gsl/gsl_randist.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_cdf.h>

//My headers
#include "structs.h"
#include "netex.h"
#include "files.h"

//The main function
void main(int argc, char **argv) {
	//Say hi to user
	printf("Hello! This script is an example implementation of my neural net library.\n");
	printf("It trains a small neural network to recognize handwritten digits from the MNIST database.\n");
	printf("Check it out!\n");
	printf("==========================================================================================\n\n");
	
	//Declare variables
		//Iteration variables
	int g, h, i, j, k;
	
		//Ints for looping training/testing + storing label
	int epochs, blocks, block_count, test_count, label, out_size, count;
	int out_weight_size1, out_weight_size2;
	int *nnum;
	
	char def_img[] = "MNIST_CSV/mnist_test.csv";
	char train_img[] = "MNIST_CSV/mnist_train.csv";
	char *imgin = def_img;
	char *imgtrain = train_img;

	char *input;
	bool oneShot = false;

	epochs = 50;
	blocks = 50000;
	block_count = 1;
	test_count = 1000;

	//Float for holding time elapsed while training, plus one for holding cost of example cost & avergae cost
	double time, cost, avgcst, dn, regco, lambda;
	
	/* DEBUG.
	 * BIT1 = Printout the image after conversion to float
	 * BIT2 = Printout the expected output vector
	 * BIT3 = Print output after feedforward
	 * BIT4 = Printout the gradient of output layer after backprop
	 * BIT5 = Printout the average gradient of the output layer
	 * BIT7 = Load in settings file.
	 * BIT8 = Enable training.
	 * BIT9 = Save settings after each epoch
	 */
	int debug = 0;

	FILE *config = fopen("tconfig.yml", "r");

	//Go through the file, get settings
		//Debug settings
	int arg;
	int argsize;
	while ((arg = getopt(argc, argv, "f:s:p")) != -1) {
		argsize = strlen(optarg) + 1;
		switch (arg) {
			case 'f':
				debug |= BIT7;
				input = malloc(argsize);

				strcpy(input, optarg);
				break;
			case 's':
				oneShot = true;
				imgin = malloc(argsize);

				strcpy(imgin, optarg);
				break;
			break;
		}
	}

	char *key = get_yaml("print_img", config);
	if (strtobool(key)) {
		debug |= BIT1;
		printf("Printing input images as float array...\n");
	}
	free(key);

	key = get_yaml("expected_out", config);
	if (strtobool(key)) {
		debug |= BIT2;
		printf("Printing the expected output for each image...\n");
	}
	free(key);

	key = get_yaml("print_output", config);
	if (strtobool(key)) {
		debug |= BIT3;
		printf("Printing the output of the network for each image...\n");
	}
	free(key);

	key = get_yaml("output_gradient", config);
	if (strtobool(key)) {
		debug |= BIT4;
		printf("Printing the gradient of the output layer each example...\n");
	}
	free(key);

	key = get_yaml("output_avg_grad", config);
	if (strtobool(key)) {
		debug |= BIT5;
		printf("Printing the average gradient of the output layer...\n");
	}
	free(key);

	key = get_yaml("enable_training", config);
	if (strtobool(key)) {
		debug |= BIT8;
		printf("Training enabled...\n");
	}
	//Print a warning message if training is disabled
	else {
		printf("WARNING: Training is not enabled! The network will only be tested.\n");
	}
	free(key);

	key = get_yaml("save_network", config);
	if (strtobool(key)) {
		debug |= BIT9;
		printf("Network settings will be saved after each epoch.\n");
	}
	else {
		printf("NOT saving settings to file...\n");
	}
	free(key);

		//Get architecture settings
	key = get_yaml("meta-settings", config);
	//If we found the label 'meta-settings', then look for the settings.
	if (strtobool(key)) {
		key = get_yaml("epochs", config);
		epochs = atoi(key);
		free(key);

		key = get_yaml("blocks", config);
		blocks = atoi(key);
		free(key);

		key = get_yaml("block_count", config);
		block_count = atoi(key);
		free(key);

		key = get_yaml("test_count", config);
		test_count = atoi(key);
		free(key);
		
		key = get_yaml("dn", config);
		dn = atof(key);
		free(key);

		key = get_yaml("lambda", config);
		lambda = atof(key);
		free(key);


		key = get_yaml("layers", config);
		if (strtobool(key)) {
			key = get_yaml("count", config);
			count = atoi(key);
			int countbyte = count * sizeof(int);
			nnum = malloc(countbyte);
			
			char spec[5];
			for (int i = 0; i < 5; i++) {
				spec[i] = '\0';
			}
			free(key);

			spec[0] = 'L';

			for (int i = 0; i < count; i++) {
				sprintf(spec + 1, "%d", i + 1);

				key = get_yaml(spec, config);
				nnum[i] = atoi(key);
			}	
		}
	}
	else {
		printf("No architecture settings found, using defaults...\n");
		nnum = malloc(3 * sizeof(int));
		nnum[0] = 784;
		nnum[1] = 30;
		nnum[2] = 10;

		dn = 0.025;
		lambda = 5;
	}

	fclose(config);

	//If we're just going to test a single example, then set debug properly.
	if (oneShot) {
		debug &= ~BIT8;
		debug &= ~BIT9;
		debug |= BIT7;

		epochs = 1;
		block_count = 1;
		blocks = 1;
		test_count = 1;
	}
		//Declare initialization settings
	netinit net_init;

		//Get number of layers
	const int layer_count = count;

	out_size = nnum[layer_count - 1];


		//Define initialization settings
	net_init.ncnt = &nnum[0];
	net_init.layers = layer_count;

		//Get vector array of expected outputs
	gsl_vector *corrects[10];
		//Pointer for the future
	gsl_vector *correct;

	for (i = 0; i < 10; i++) {
		corrects[i] = expected_out(i, 10);
	}
	printf("Network configuration:\n");
	printf("  Layers: %d\n", net_init.layers);
	for (int p = 0; p < net_init.layers; p++) {
		printf("    Layer %d: %d neurons\n", p, net_init.ncnt[p]);
	}
	printf("  Epochs: %d\n", epochs);
	printf("  Blocks: %d\n", blocks);
	printf("  Examples per block: %d\n", block_count);
	printf("  Testing examples: %d\n", test_count);
	printf("\n");

		//empty input vector, the input goes here before the net
	gsl_vector *in_vect = gsl_vector_calloc(net_init.ncnt[0] + 1);


	//Initialize the architecture
	layer *network = build_architecture(net_init);

	out_weight_size1 = ((network + layer_count - 1) -> ws.size1);
	out_weight_size2 = ((network + layer_count - 1) -> ws.size2);

	//Initialize gradient and average gradient.
	gradient *grad, *avg_grad;
	grad = malloc(layer_count * sizeof(gradient));
	avg_grad = malloc(layer_count * sizeof(gradient));

	for (i = 1; i < layer_count; i++) {
		(grad + i) -> wgrad = gsl_matrix_calloc(net_init.ncnt[i], net_init.ncnt[i - 1]);
		(grad + i) -> bgrad = gsl_vector_calloc(net_init.ncnt[i]);		

		(avg_grad + i) -> wgrad = gsl_matrix_calloc(net_init.ncnt[i], net_init.ncnt[i - 1]);
		(avg_grad + i) -> bgrad = gsl_vector_calloc(net_init.ncnt[i]);		
	}

	
	//Optionally load in a settings file
	if (debug & BIT7) {
		printf("Loading in a pre-trained settings file \"%s\"...\n", input);
		FILE *netf = fopen(input, "r");
		
		read_net(netf, network);

		fclose(netf);
	}
	
	//Loop through the epochs
	for (g = 0; g < epochs; g++) {
		//Print epoch number, count time since start of epoch
		clock_t start, stop;
		start = clock();
		avgcst = 0;

		printf("Epoch %d\n", g);

		//If selected, train the network for the set block number, where each block contains
		//block_count number of examples
		if (debug & BIT8) {
			//Before we loop, open the training data file
			FILE *file = fopen(imgtrain, "r");

			for (h = 0; h < blocks; h++) {
				//printf("  Block %d\n", h);
				//Clear the average gradient
				for (i = 1; i < layer_count; i++) {
					gsl_matrix_set_zero((avg_grad + i) -> wgrad);
					gsl_vector_set_zero((avg_grad + i) -> bgrad);
				}
			
				//Loop through individual examples until the end of the block
				for (i = 0; i < block_count; i++) {
					//Get the input data for this example, load it into the input layer
					get_img(debug, file, in_vect);
					
						//Copy the vector into input
					for (j = 1; j < net_init.ncnt[0]; j++) {
						double this_val = gsl_vector_get(in_vect, j);
						gsl_vector_set(&(network -> vals), j - 1, this_val);
					}

					//Get label
					label = (int) gsl_vector_get(in_vect, 0);
					if (debug & BIT3) {printf("    Example: %d   Label: %d\n", i, label);}
					//Get the expected output
					correct = corrects[label];

					//Printout the expected output
					if (debug & BIT2) {
						printf("Expected out for number %d: [", label);
						for (j = 0; j < 10; j++) {
							printf(" %.1lf ", gsl_vector_get(correct, j));
						}
						printf("]\n");
					}
					
					//Call training function. Training function should return the cost and
					//write to a pre-allocated gradient struct.
					cost = backprop(debug, net_init, network, grad, correct);

					//Make sure that your cost & gradient are always the average of the block
					avgcst += cost / blocks;
					for (i = 1; i < layer_count; i++) {
						gsl_matrix_add((avg_grad + i) -> wgrad, (grad + i) -> wgrad);
						gsl_vector_add((avg_grad + i) -> bgrad, (grad + i) -> bgrad);
					}
				}
	
	
				//Do the gradient descent based on the average gradient
					//Get regularization coefficient
				regco = 1 - (lambda * (dn / (blocks * block_count)));

					//Get dn
				dn = dn / block_count;

					//Do the descent!!
				gradient_descend(net_init, network, avg_grad, dn, regco);

				//Debug: Printout average gradient
				if (debug & BIT5) {
					double this_weight, this_bias;
					//Iterate over each row
					for (j = 0; j < out_weight_size1; j++) {
						//Iterate over each entry in the row
						printf("Avg Grad Neuron %d: [", j);
						for (k = 0; k < out_weight_size2; k++) {
							this_weight = gsl_matrix_get((avg_grad + layer_count - 1) -> wgrad, j, k);
							printf(" %.3lf ", this_weight);
						}
						this_bias = gsl_vector_get((avg_grad + layer_count - 1) -> bgrad, j);
						printf("] [ %.3lf ]\n", this_bias);
					}
				}
			}	
			//Close the file
			fclose(file);
		}

		//Check how long the training took
		stop = clock();
		time = (double)(stop - start) / CLOCKS_PER_SEC;

		printf("  Time: %.3lfsec\n", time);

		//At the end of this epoch, test the network. Loop through preset number of
		//examples & get the score.
		FILE *tfile = fopen(imgin, "r");

		int right, output;
		double accuracy;

		right = 0;

		for (h = 0; h < test_count; h++) {
			//Get the input data for this example, load it into the input layer
			get_img(debug, tfile, in_vect);

			
			//Copy the vector into input
			for (j = 1; j < net_init.ncnt[0]; j++) {
				double this_val = gsl_vector_get(in_vect, j);
				gsl_vector_set(&(network -> vals), j - 1, this_val);
			}

			//Get label
			label = (int) gsl_vector_get(in_vect, 0);
			//Get the expected output
			correct = corrects[label];

			//Call testing function. Testing function should return
			//the accuracy of the network as a float.
			output = test(debug, net_init, network, correct);
			if (output) {
				right++;
			}

		}
		accuracy = (float) right / (float) test_count;
		accuracy *= 100;
		//Show the user the average accuracy over testing session
		printf("  Correct: %d Incorrect: %d \n  Score: %.3lf Cost: %.3lf\n", right, test_count - right, accuracy, avgcst);
		fclose(tfile);

		//Optionally save the weights and biases after a single epoch
			//Only save if saving is enabled & training is enabled
		if ((debug & BIT8) && (debug & BIT9)) {
			printf("Saving settings...\n");
			FILE *outf = fopen("net.txt", "w+");
			//Iterate over each layer except input layer.
			for (int i = 1; i < layer_count; i++) {
				//Iterate over each neuron in this layer.
				for (j = 0; j < net_init.ncnt[i]; j++) {
					//Iterate over each neuron in the last layer
					for (k = 0; k < net_init.ncnt[i - 1]; k++) {
						//Print the weight k in row j
						fprintf(outf, "%lf,", gsl_matrix_get(&((network + i) -> ws), j, k));
					}
					fprintf(outf, "|%lf\n", gsl_vector_get(&((network + i) -> bias), j));
				}
				fprintf(outf, "L\n", i + 1);
			}
		
			//Close the file
			fclose(outf);
		}

	}
	free(nnum);
}
