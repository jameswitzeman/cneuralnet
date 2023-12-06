//This script provides functionality for taking in a .csv file and loading it line by line.
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <gsl/gsl_math.h>
#include "files.h"
#include "netex.h"

//This is the longest a single line can be. Should be kept reasonably large.
#define MAX_CHARS 0xFFFF

//Like a scan line, but it spits out nice formatted arrays.
void get_img(int debug, FILE *fp, gsl_vector *empty_vect) {
	int asize = empty_vect -> size;
	int i, j, intout[asize];
	double flout;

	//Small string buffer for getting the numbers
	char num[3], gotc;

	//Now. The string contains numbers and commas. We need to keep a string
	//buffer of 3 characters to hold the number. Then we convert it into a number.
	gotc = 1;
	//Here, j keeps track of position in num[],
	//i keeps track of position in intout[].
	i = j = 0;
	while ((gotc != '\n')) {
		gotc = fgetc(fp);
		if (gotc != ',') {
			num[j] = gotc;
			j++;
		}


		else {
			intout[i] = atoi(num);
			for (int l = 0; l < 3; l++) {
				num[l] = '\0';
			}
			j = 0;
			i++;
		}
		if (feof(fp)) {
			gsl_vector_set(empty_vect, 0, 256);
		}
	}

	//Before finishing up, make sure we save the very lasy entry.
	for (int l = 0; l < 3; l++) {
		if (num[l] == '\n') {
			num[l] = '\0';
		}

	}
	intout[asize - 1] = atoi(num);


	//Now we should have an int array full of our image pixels as entries 0-255. We need to
	//convert this into floats between 0-1. I'll use a sigmoid for this, if the net is having
	//issues try using a different one!
	//Make sure we keep the label intact
	flout = (double) intout[0];

	gsl_vector_set(empty_vect, 0, flout);
	for (i = 1; i < asize; i++) {
		flout = linear_map(intout[i], 0, 255, 0, 1);
		gsl_vector_set(empty_vect, i, flout);
		if (debug & BIT1) {
			printf("Pixel %d DValue: %d FValue: %.4lf\n", i, intout[i], flout);
		}
	}
}

//This function should read in settings for a previously
//trained network. Note that the structure of the network
//must be known beforehand.
void read_net(FILE *fp, layer *lays) {

	//i is the layer, j is the neurons from layer i, k is neurons from layer l - 1 AKA columns
	int i, j, k, if_bias, strnum;
	layer *thislay;
	gsl_vector *thisbias;
	gsl_matrix *thismatrix;
	double thisnum;
	char gotc;
	char thisd[12] = "\0\0\0\0\0\0\0\0\0\0\0\0";

	i = 1;
	k = strnum = if_bias = 0;
	j = 1;
	gotc = getc(fp);
	while (gotc != EOF) {
		//Get a simple pointer for the layer, weight, and bias
		thislay = (lays + i);
		thisbias = &(thislay -> bias);
		thismatrix = &(thislay -> ws);

		//Newline signifies the next row of the matrix, newline plus L is a new layer
		if (gotc == '\n') {
			j++;
			strnum = k = 0;
		}
		else if (gotc == 'L') {
			i ++;
			k = j = 0;
		}
		else if (gotc == '|') {
			if_bias = 1;
		}
		else if (gotc == ',') {
			strnum = 0;

			k++;

			thisnum = atof(thisd);
			
			//Check if this number should be a bias or weight
			if (if_bias) {
				gsl_vector_set(thisbias, j - 1, thisnum);
			}
			else {
				gsl_matrix_set(thismatrix, j - 1, k - 1, thisnum);
			}

			//We've hit the end of this float, clear the buffer
			for (int n = 0; n < 12; n++) {
				thisd[n] = "\0";
			}
			if_bias = 0;
		}
		else {
			thisd[strnum] = gotc;
			strnum++;
		}
		gotc = getc(fp);
	}	
}

char *get_yaml(char *label, FILE *file){
	bool match = false;
	bool done = false;
	bool isKey = false;
	bool isComment = false;

	//File input buffer
	char *buffer = malloc(50);
	char thischar = fgetc(file);

	int buffer_place = 0;
	int got_eof = 0;

	for (int i = 0; i < 50; i++) {
		buffer[i] = '\0';
	}

	while (!done && (thischar != EOF)) {
		//First check for colon; this means we should check if the buffer matches
		
		if (thischar == ':') {
			isKey = true;

			//Check if the strings match
			int j = 0;
			match = true;
			while (buffer[j] != '\0' && label[j] != '\0' && match == true) {
				if (buffer[j] != label[j]) {
					match = false;
				}

				j++;
			}

			//Clear buffer
			for (int i = 0; i < 50; i++) {
				buffer[i] = '\0';
			}
			buffer_place = 0;

		}

		else if (thischar == '\n') {
			//This signals the end of a comment if we detected a '#'. Check for that
			if (isComment) {
				isComment = false;
			}

			//We reached the end of the key, so the key should
			//be in the buffer. If it's the right key, then return it
			if (isKey && match) {
				//If the buffer has a tab or space or null, then just
				//return 'true', indicating that we found the label
				if (buffer[0] == '\0' || buffer[0] == ' ' || buffer[0] == '\n') {
					return "true";
				}
				
				return buffer;
			}

			//Clear buffer
			for (int i = 0; i < 50; i++) {
				buffer[i] = '\0';
			}
			buffer_place = 0;
			isKey = false;
			match = false;
		}

		else if (thischar == '#') {
			isComment = true;
		}

		else if (thischar != ' ' && !isComment) {
			buffer[buffer_place] = thischar;
			buffer_place++;
		}
		
		else if (thischar == EOF) {
			rewind(file);
			got_eof++;
		}
		thischar = fgetc(file);
	}
	return buffer;

}

bool strtobool(char *string) {
	char yes[] = "true";

	int i = 0;
	while (string[i] != '\0') {
		if (yes[i] == '\0') {
			return false;
		}

		else if (yes[i] != string[i]) {
			return false;	
		}	

		i++;
	}
	return true;
}
