#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

__global__ void gpu_matrix_mult(float *a,float *b, float *c, 
    int m, int n, int k)
{ 
    int row = blockIdx.y * blockDim.y + threadIdx.y; 
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    float sum = 0;

    if( col < k && row < m) 
    {
        for(int i = 0; i < n; i++) 
        {
            sum += a[row * n + i] * b[i * k + col];
        }
        c[row * k + col] = sum;
    }
} 

__global__ void gpu_matrix_transpose(float *a, float *b) 
{
    int rows = sizeof(a);
    int cols = sizeof(a[0]);

    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < cols && idy < rows) 
    {
        b[idx * rows + idy] = a[idy * cols + idx];
    }
}