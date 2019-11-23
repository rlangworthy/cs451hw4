/* Matrix normalization.
 * Compile with "gcc matrixNorm.c"
 */

 #include <stdio.h>
 #include <stdlib.h>
 #include <time.h>
 #include <sys/time.h>
 #include <math.h>
 
 /* Program Parameters */
 #define N 6000  /* Matrix size */
 
 #define BLOCK_SIZE 64


 /* Matrices */
volatile float A[N][N], B[N][N];
float h_a[N][N], h_b[N][N];
 
 
 /* Initialize A and B*/
 void initialize_inputs() {
     int row, col;
     
     srand((unsigned)time(NULL));
     for (row = 0; row < N; row++) {
         for (col = 0; col < N; col++) {
             A[row][col] = (float)rand() / 32768.0;
             h_a[row][col] = A[row][col];
             B[row][col] = 0.0;
             h_b[row][col] = 0.0;

         }
     }
     
 }
 
 
 /* Kernel function */
 
 __global__ void matrixNorm(float *A, float *B, int n) {
    int col = blockIdx.x;
    int row, stride;
    int tid = threadIdx.x;
    float mu, sigma, partial=0; // Mean and Standard Deviation
    __shared__ float partials[BLOCK_SIZE], fullCol[N];

    //set up partial sums and copy working column into shared memory
    for(row = threadIdx.x; row < n; row += blockDim.x){
        fullCol[row] = A[row*n + col];
        partial += fullCol[row];
    }
    partials[tid] = partial;
    __syncthreads();
    //reduction for sum
    for (stride = 1; stride < blockDim.x; stride *= 2) {
        if (tid % (2*stride) == 0){
            partials[tid] += partials[tid+stride];
        }
        __syncthreads();
    }
    //calculate mu, reset partial
    mu = partials[0]/n;
    partial = 0;


    //repeat for sigma
    for(row = threadIdx.x; row < n; row += blockDim.x){
        partial += powf(fullCol[row]-mu, 2.0);
    }
    partials[tid] = partial;
    __syncthreads();
    //reduction for variance * n
    for (stride = 1; stride < blockDim.x; stride *= 2) {
        if (tid % (2*stride) == 0){
            partials[tid] += partials[tid+stride];
        }
        __syncthreads();
    }
    //calculate mu
    sigma = partials[0]/n;
    sigma = sqrt(sigma);

    //use copied column to fill in B array
    for(row = threadIdx.x; row < n; row += blockDim.x){
        if (sigma == 0.0){
            B[row*n + blockIdx.x] = 0.0;
        }
        else{
            B[row*n + blockIdx.x] = (fullCol[row] -mu) / sigma;
        }
    }

}
 

void matrixNormSerial() {
    int row, col;
    float mu, sigma; // Mean and Standard Deviation
    
    printf("Computing Serially.\n");
    
    for (col=0; col < N; col++) {
        mu = 0.0;
        for (row=0; row < N; row++)
            mu += A[row][col];
        mu /= (float) N;
        sigma = 0.0;
        for (row=0; row < N; row++)
            sigma += powf(A[row][col] - mu, 2.0);
        sigma /= (float) N;
        sigma = sqrt(sigma);

        for (row=0; row < N; row++) {
            if (sigma == 0.0)
                B[row][col] = 0.0;
            else
                B[row][col] = (A[row][col] - mu) / sigma;
        }
    }
    
}
 
 
 int main(int argc, char **argv) {
     /* Timing variables */
     struct timeval start, stop;  /* Elapsed times using gettimeofday() */
     struct timezone tzdummy;
     unsigned long long runtime;
     



     /* Initialize A and B */
     initialize_inputs();
     
     
    // Allocate memory space on the device
    float *d_a, *d_b;
    cudaMalloc((void **) &d_a, sizeof(float)*N*N);
    cudaMalloc((void **) &d_b, sizeof(float)*N*N);

    // copy matrix A from host to device memory
    cudaMemcpy(d_a, h_a, sizeof(float)*N*N, cudaMemcpyHostToDevice);

    // some events to count the execution time
    cudaEvent_t cstart, cstop;
    cudaEventCreate(&cstart);
    cudaEventCreate(&cstop);
    float gpu_elapsed_time_ms;

    dim3 dimGrid(N, 1, 1);
    dim3 dimBlock(BLOCK_SIZE, 1,1);


     /* Start Clock */
     printf("\n---------------------------------------------\n");
     printf("Matrix size N = %d", N);
     printf("\nStarting clock.\n\n");
     gettimeofday(&start, &tzdummy);
    
    
    // Launch simple matrix multiplication kernel
    matrixNormSerial();

    /* Stop Clock */
    gettimeofday(&stop, &tzdummy);
    runtime = (unsigned long long)(stop.tv_sec - start.tv_sec) * 1000000 + (stop.tv_usec - start.tv_usec); 

     
     /* Display timing results */
     printf("Runtime = %g ms.\n", (float)runtime/(float)1000);
     printf("\nStopped clock.");
     printf("\n---------------------------------------------\n");
    

     /* Start Clock */
     printf("\n---------------------------------------------\n");
     printf("Matrix size N = %d", N);
     printf("\nStarting Cuda clock.\n\n");
     cudaEventRecord(cstart, 0);

     matrixNorm<<<dimGrid, dimBlock>>>(d_a, d_b, N);  
    // start to count execution time of GPU Kernel 
     cudaEventRecord(cstop, 0);
     cudaEventSynchronize(cstop);
 
     // Transfer results from device to host
     cudaMemcpy(h_b, d_b, sizeof(float)*N*N, cudaMemcpyDeviceToHost);
    // compute time elapse on GPU computing
    cudaEventElapsedTime(&gpu_elapsed_time_ms, cstart, cstop);
    printf("Time elapsed on matrix norm on GPU: %f ms.\n\n", gpu_elapsed_time_ms);
    printf("Runtime = %g ms.\n", (float)gpu_elapsed_time_ms);
    printf("\nStopped clock.");
    printf("\n---------------------------------------------\n");

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFreeHost(h_a);
    cudaFreeHost(h_b);

    int i;
    printf("Spot check for correctness on row 100, cols 0-9: \n");
    for(i=0; i < 10; i++){
        printf("B: %5.2f  b_h: %5.2f\n", B[100][i], h_b[100][i]);
    }


     exit(0);
 }