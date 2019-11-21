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
 
 #define BLOCK_SIZE 16
 #define TILE_WIDTH 16


 /* Matrices */
float A[N][N], B[N][N];
 
 
 /* Initialize A and B*/
 void initialize_inputs() {
     int row, col;
     
     srand((unsigned)time(NULL));
     for (row = 0; row < N; row++) {
         for (col = 0; col < N; col++) {
             A[row][col] = (float)rand() / 32768.0;
             B[row][col] = 0.0;
         }
     }
     
 }
 
 
 /* Kernel function */
 
 __global__ void matrixNorm(float *A, float *B, int n) {
    //int col = blockIdx.x;
    int row, stride;
    int tid = threadIdx.x;
    float mu, sigma, partial=0; // Mean and Standard Deviation
    __shared__ float partials[16], fullCol[N];

    //set up partial sums and copy working column into shared memory
    for(row = threadIdx.x; row < n; row += blockDim.x){
        fullCol[row] = A[threadIdx.x*n + blockIdx.x];
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
            B[threadIdx.x*n + blockIdx.x] = 0.0;
        }
        else{
            B[threadIdx.x*n + blockIdx.x] = (fullCol[row] -mu) / sigma;
        }
    }

}
 
 
 
 int main(int argc, char **argv) {
     /* Timing variables */
     //struct timeval start, stop;  /* Elapsed times using gettimeofday() */
     //struct timezone tzdummy;
     //unsigned long long runtime;
     



     /* Initialize A and B */
     initialize_inputs();
     
     
    // Allocate memory space on the device
    float *d_a, *d_b;
    cudaMalloc((void **) &d_a, sizeof(float)*N*N);
    cudaMalloc((void **) &d_b, sizeof(float)*N*N);

    // copy matrix A from host to device memory
    cudaMemcpy(d_a, A, sizeof(float)*N*N, cudaMemcpyHostToDevice);

    // some events to count the execution time
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float gpu_elapsed_time_ms;

    dim3 dimGrid(N, 1, 1);
    dim3 dimBlock(16, 1,1);


     /* Start Clock */
     printf("\n---------------------------------------------\n");
     printf("Matrix size N = %d", N);
     printf("\nStarting clock.\n\n");
     
     // start to count execution time of GPU Kernel 
    cudaEventRecord(start, 0);
    
    // Launch simple matrix multiplication kernel
    matrixNorm<<<dimGrid, dimBlock>>>(d_a, d_b, N);  
    
    // time counting terminate
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    // Transefr results from device to host
    cudaMemcpy(B, d_b, sizeof(float)*N*N, cudaMemcpyDeviceToHost);
     
   
    // compute time elapse on GPU computing
    cudaEventElapsedTime(&gpu_elapsed_time_ms, start, stop);
    printf("Time elapsed on simple matrix multiplication on GPU: %f ms.\n\n", gpu_elapsed_time_ms);

     
     
     /* Display timing results */
     printf("Runtime = %g ms.\n", (float)gpu_elapsed_time_ms);
     printf("\nStopped clock.");
     printf("\n---------------------------------------------\n");
     
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFreeHost(A);
    cudaFreeHost(B);



     exit(0);
 }