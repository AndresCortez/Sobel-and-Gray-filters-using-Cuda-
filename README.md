# Sobel-and-Gray-filters-using-Cuda-
Compute the Gray Scale and the Sobel Operator of an image by using CUDA and OpenCV 
It is necessary to have OpenCV and CUDA to run the program
In order to compile: 
nvcc FinalProject.cu `pkg-config --cflags --libs opencv`  
To run:
./a.out DesiredImage
Ex: 
./a.out Lenna.png
