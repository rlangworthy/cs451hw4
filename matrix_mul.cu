/*
file name: matrix_mul.cu
 *
 *  matrix.cu contains two implemention of matrix multiplication in class
 *  Each matrix size is 1024*1024
 *  In this program, the elapesed time is only calculating kernel time.  Time periods of allocating  cuda memory,  data transfer and freeing cuda memory are not included. However, in your homework, you should include these overheads.
 *
 */
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

#define BLOCK_SIZE 16
#define TILE_WIDTH 16
/*
*********************************************************************
function name: gpu_matrix_mult
description: simple impliementation
*********************************************************************
*/
__global__ void gpu_matrix_mult(float *A, float *B, float *C,  int n)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
        if (row < n && col < n) {
            for (int i = 0; i < n; ++i) {
                C[row * n + col] += A[row * n + i] * B[i * n + col];
      }
    }
}


/*
*********************************************************************
function name: MatrixMul_tileKernel
description: Using tiling stratagy for matrix multiplication in GPU
*********************************************************************
*/


__global__ void MatrixMul_tileKernel(float* Md, float* Nd, float* Pd, int Width){

int Row = blockIdx.y*TILE_WIDTH + threadIdx.y;
int Col = blockIdx.x*TILE_WIDTH + threadIdx.x;
int tx = threadIdx.x, ty = threadIdx.y;
__shared__ float a[TILE_WIDTH][TILE_WIDTH], b[TILE_WIDTH][TILE_WIDTH];
float Pvalue = 0;
//Each thread computes one element of the block sub-matrix
for(int k=0; k< Width/TILE_WIDTH; k++){
    a[ty][tx] = Md[Row*Width+k*TILE_WIDTH+tx];
    b[ty][tx] = Nd[Col+Width*(k*TILE_WIDTH + ty)];

    __syncthreads(); //sync all threads in a block;
    for(int kk=0; kk<TILE_WIDTH; kk++)
    Pvalue += a[ty][kk]*b[kk][tx];

    __syncthreads(); //avoid memory hazards;
}
Pd[Row*Width+Col] = Pvalue;
}

/*
*********************************************************************
function name: main
description: test and compare
parameters:
            none
return: none
*********************************************************************
*/
int main(int argc, char const *argv[])
{
    
    int n=1024;
    /* Fixed seed for illustration */
    srand(3333);
    // allocate memory in host RAM
    float *h_a, *h_b, *h_c;
    cudaMallocHost((void **) &h_a, sizeof(float)*n*n);
    cudaMallocHost((void **) &h_b, sizeof(float)*n*n);
    cudaMallocHost((void **) &h_c, sizeof(float)*n*n);
    //generate matrix A and B
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            h_a[i * n + j] = rand() % 1024/2.3;
            h_b[i * n + j] = rand() % 24/3.3;
        }
    }


    float gpu_elapsed_time_ms;

    // some events to count the execution time
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

   

    // Allocate memory space on the device
    float *d_a, *d_b, *d_c;
    cudaMalloc((void **) &d_a, sizeof(float)*n*n);
    cudaMalloc((void **) &d_b, sizeof(float)*n*n);
    cudaMalloc((void **) &d_c, sizeof(float)*n*n);

    // copy matrix A and B from host to device memory
    cudaMemcpy(d_a, h_a, sizeof(float)*n*n, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, sizeof(float)*n*n, cudaMemcpyHostToDevice);

    unsigned int grid_rows = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_cols, grid_rows);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);

    
    // start to count execution time of GPU Kernel 
    cudaEventRecord(start, 0);
    // Launch simple matrix multiplication kernel
    gpu_matrix_mult<<<dimGrid, dimBlock>>>(d_a, d_b, d_c, n);  
    // time counting terminate
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    // Transefr results from device to host
    cudaMemcpy(h_c, d_c, sizeof(float)*n*n, cudaMemcpyDeviceToHost);
     
   
    // compute time elapse on GPU computing
    cudaEventElapsedTime(&gpu_elapsed_time_ms, start, stop);
    printf("Time elapsed on simple matrix multiplication on GPU: %f ms.\n\n", gpu_elapsed_time_ms);

    
    cudaEventRecord(start, 0);
    // Launch tile matrix multiplication kernel
    MatrixMul_tileKernel<<<dimGrid, dimBlock>>>( d_a, d_b, d_c, n); 
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    // Transefr results from device to host
    cudaMemcpy(h_c, d_c, sizeof(float)*n*n, cudaMemcpyDeviceToHost);
     
    // compute time elapse on GPU kernel
    cudaEventElapsedTime(&gpu_elapsed_time_ms, start, stop);
    printf("Time elapsed on  matrix multiplication with tiling strategy on GPU: %f ms.\n\n", gpu_elapsed_time_ms);
    // free memory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    cudaFreeHost(h_a);
    cudaFreeHost(h_b);
    cudaFreeHost(h_c);
    return 0;
}
