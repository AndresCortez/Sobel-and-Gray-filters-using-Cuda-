/* This code will generate a Sobel image and a Gray Scale image. Uses OpenCV, to compile:
   nvcc FinalProject.cu `pkg-config --cflags --libs opencv`  

   Copyright (C) 2018  Jose Andres Cortez Villao

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.*/
	
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <opencv/highgui.h>
#include "utils/cheader.h"


typedef enum color {BLUE, GREEN, RED} Color;	//Constants that contains the values for each color of the image 

/*The gray function obtain an average of each pixel and assigned to the correct position in the array using 
Channels and step constants*/
__global__ void gray(unsigned char *src, unsigned char *dest, int width, int heigth, int step, int channels) { 
	int ren, col;
	float r, g, b;
	
	ren = blockIdx.x; // Variables that parallelize the code 
	col = threadIdx.x;
	r = 0; g = 0; b = 0;

	r += (float) src[(ren * step) + (col * channels) + RED];
	g += (float) src[(ren * step) + (col * channels) + GREEN];
	b += (float) src[(ren * step) + (col * channels) + BLUE];

	dest[(ren * step) + (col * channels) + RED] =  (unsigned char) ((r+g+b)/3);
	dest[(ren * step) + (col * channels) + GREEN] = (unsigned char) ((r+g+b)/3);
	dest[(ren * step) + (col * channels) + BLUE] = (unsigned char) ((r+g+b)/3);
}
/*The sobel function uses a convolution algorithm to obtain the edges of the image */

__global__ void sobel(unsigned char *src, unsigned char *dest, int width, int heigth, int step, int channels){
	int i, j;
	int ren, col, tmp_ren, tmp_col;
	int gx[3][3]={{-1,0,1},{-2,0,2},{-1,0,1}}; // gx is defined in the Sobel algorithm
	int gy[3][3]={{1,2,1},{0,0,0},{-1,-2,-1}}; // gy is defined in the Sobel algorithm 
	char temp[3][3];
	 
	ren = blockIdx.x;
	col = threadIdx.x;

	tmp_ren = 0;
	tmp_col = 0;
	
	//Multiplication of the 3x3 matrix for each color 
	for (i = -1; i < 2; i++) {
		for (j = -1; j < 2; j++) {
			temp[i+1][j+1]=(int) src[(ren * step) + (col * channels) + RED + i + 1];
			tmp_ren=tmp_ren + temp[i+1][j+1]*gx[i+1][j+1];
			tmp_col=tmp_col + temp[i+1][j+1]*gy[i+1][j+1];				
		}
	}
	dest[(ren * step) + (col * channels) + RED] =  (unsigned char) sqrtf(tmp_col*tmp_col+tmp_ren*tmp_ren);;

	tmp_ren = 0;
	tmp_col = 0;
	for (i = -1; i < 2; i++) {
		for (j = -1; j < 2; j++) {
			temp[i+1][j+1]=(int) src[(ren * step) + (col * channels) + GREEN + i + 1];
			tmp_ren=tmp_ren + temp[i+1][j+1]*gx[i+1][j+1];
			tmp_col=tmp_col + temp[i+1][j+1]*gy[i+1][j+1];				
		}
	}
	dest[(ren * step) + (col * channels) + GREEN] =  (unsigned char) sqrtf(tmp_col*tmp_col+tmp_ren*tmp_ren);;


	tmp_ren = 0;
	tmp_col = 0;
	for (i = -1; i < 2; i++) {
		for (j = -1; j < 2; j++) {
			temp[i+1][j+1]=(int) src[(ren * step) + (col * channels) + BLUE + i + 1];
			tmp_ren=tmp_ren + temp[i+1][j+1]*gx[i+1][j+1];
			tmp_col=tmp_col + temp[i+1][j+1]*gy[i+1][j+1];				
		}
	}
	dest[(ren * step) + (col * channels) + BLUE] =  (unsigned char) sqrtf(tmp_col*tmp_col+tmp_ren*tmp_ren);
}

int main(int argc, char* argv[]) {
	int i, step, size;
	double acum; 
	unsigned char *dev_src, *dev_gray,*dev_sobel;
		
	if (argc != 2) {
		printf("usage: %s source_file\n", argv[0]);
		return -1;
	}
	//Obtain and create the image using OpenCV 
	IplImage *src = cvLoadImage(argv[1], CV_LOAD_IMAGE_COLOR);
	IplImage *grayImage = cvCreateImage(cvSize(src->width, src->height), IPL_DEPTH_8U, 3);
	IplImage *sobelImage = cvCreateImage(cvSize(src->width, src->height), IPL_DEPTH_8U, 3);
	
	//Check if there is an image as an argument
	if (!src) {
		printf("Could not load image file: %s\n", argv[1]);
		return -1;
	}
	
	size = src->width * src->height * src->nChannels * sizeof(uchar);
	cudaMalloc((void**) &dev_src, size);
	cudaMalloc((void**) &dev_gray, size);
	cudaMalloc((void**) &dev_sobel, size);
	cudaMemcpy(dev_src, src->imageData, size, cudaMemcpyHostToDevice);
	
	acum = 0;
	step = src->widthStep / sizeof(uchar);
	
	//Compute the execution time for each function 
	printf("Starting...\n");
	for (i = 0; i < N; i++) {
		start_timer();
		gray<<<src->height, src->width>>>(dev_src, dev_gray, src->width, src->height, step, src->nChannels);
		cudaMemcpy(grayImage->imageData, dev_gray, size, cudaMemcpyDeviceToHost);
		acum += stop_timer();
	}
	
	for (i = 0; i < N; i++) {
		start_timer();
		cudaMemcpy(dev_gray, grayImage->imageData, size, cudaMemcpyHostToDevice);
		sobel<<<grayImage->height, grayImage->width>>>(dev_gray, dev_sobel, src->width-1, src->height-1, step, src->nChannels);
		cudaMemcpy(sobelImage->imageData, dev_sobel, size, cudaMemcpyDeviceToHost);
		acum += stop_timer();
	}
	//Free the memory of the GPU 
	cudaFree(dev_gray);
	cudaFree(dev_src);
	cudaFree(dev_sobel);
	
	printf("avg time = %.5lf ms\n", (acum / (N*2)));
	
	cvShowImage("(Original)", src);
	cvShowImage("(Gray)", grayImage);
	cvShowImage("(Sobel)", sobelImage);
	cvWaitKey(0);
	cvWaitKey(0);
	cvDestroyWindow("Lenna (Original)");
	cvDestroyWindow("Lenna (Gray)");
	cvDestroyWindow("Lenna (Sobel)");

	return 0;
}
