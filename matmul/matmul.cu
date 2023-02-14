//
// Created by hanchiao on 2/7/23.
//

#include <cstdio>
#include <cmath>
#include "error.cuh"

#define BLOCK_SIZE 16

void cpu_matrix_mult(int *h_a, int *h_b, int *h_result, int m, int n, int k){
    for(int i = 0; i < m; i++){
        for(int j = 0; j < k; j++){
            int tmp = 0;
            for(int l = 0; l < n; l++){
                tmp += h_a[i * n + l] * h_b[l * k + j];
            }
            h_result[i * k + j] = tmp;
        }
    }
}

__global__ void gpu_matrix_mult(int *d_a, int *d_b, int *d_result, int m, int n, int k){
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int sum = 0;
    if( col < k && row < m)
    {
        for(int i = 0; i < n; i++)
        {
            sum += d_a[row * n + i] * d_b[i * k + col];
        }
        d_result[row * k + col] = sum;
    }
}

int main(void){
    int m=100;
    int n=100;
    int k=100;

    int *h_a, *h_b, *h_c, *h_cc;
    CHECK(cudaMallocHost((void **)&h_a, sizeof(int) * m * n));
    CHECK(cudaMallocHost((void **)&h_b, sizeof(int) * n * k));
    CHECK(cudaMallocHost((void **)&h_c, sizeof(int) * m * k));
    CHECK(cudaMallocHost((void **)&h_cc, sizeof(int) * m * k));

    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            h_a[i * n + j] = rand() % 1024;
        }
    }

    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < k; ++j) {
            h_b[i * k + j] = rand() % 1024;
        }
    }

    int *d_a, *d_b, *d_c;
    CHECK(cudaMalloc((void **)&d_a, sizeof(int) * m * n));
    CHECK(cudaMalloc((void **)&d_b, sizeof(int) * n * k));
    CHECK(cudaMalloc((void **)&d_c, sizeof(int) * m * k));

    // copy matrix A and B from host to device memory
    CHECK(cudaMemcpy(d_a, h_a, sizeof(int)*m*n, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_b, h_b, sizeof(int)*n*k, cudaMemcpyHostToDevice));

    unsigned int grid_rows = (m + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (k + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_cols, grid_rows);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);

    gpu_matrix_mult<<<dimGrid, dimBlock>>>(d_a, d_b, d_c, m, n, k);

    CHECK(cudaMemcpy(h_c, d_c, sizeof(int)*m*k, cudaMemcpyDeviceToHost));
    //cudaThreadSynchronize();

    cpu_matrix_mult(h_a, h_b, h_cc, m, n, k);

    int ok = 1;
    for (int i = 0; i < m; ++i)
    {
        for (int j = 0; j < k; ++j)
        {
            if(fabs(h_cc[i*k + j] - h_c[i*k + j])>(1.0e-10))
            {
                ok = 0;
            }
        }
    }

    if(ok)
    {
        printf("Pass!!!\n");
    }
    else
    {
        printf("Error!!!\n");
    }

    // free memory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    cudaFreeHost(h_a);
    cudaFreeHost(h_b);
    cudaFreeHost(h_c);
    cudaFreeHost(h_cc);
    return 0;
}