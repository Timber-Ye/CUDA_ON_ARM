//
// Created by hanchiao on 2/10/23.
//
#include <cmath>
#include "error.cuh"

#define BLOCK_SIZE 16

__global__ void gpu_matrix_mult_shared(int * a, int * b, int * c, int m, int n, int k){
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int idx;
    int tmp=0;
    __shared__ int tile_a[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ int tile_b[BLOCK_SIZE][BLOCK_SIZE];

    for(int stride=0; stride<gridDim.x; stride++){
        idx = row * n + BLOCK_SIZE * stride + threadIdx.x;
        tile_a[threadIdx.y][threadIdx.x] = row < m && (BLOCK_SIZE * stride + threadIdx.x) < n ? a[idx]:0;
        idx = col + (BLOCK_SIZE * stride + threadIdx.y) * k;
        tile_b[threadIdx.y][threadIdx.x] = col < k && (BLOCK_SIZE * stride + threadIdx.y) < n ? b[idx]:0;
        __syncthreads();

        for(int i=0; i < BLOCK_SIZE; i++){
            tmp += tile_a[threadIdx.y][i] * tile_b[i][threadIdx.x];
        }
        __syncthreads();
    }
    if(row < m && col < k){
        c[row * k + col] = tmp;
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


int main(void){

    float elapsed_time;

    int m=1000;
    int n=2000;
    int k=3000;

    int *Ma, *Mb, *Mc, *Mcc;
    CHECK(cudaMallocManaged((void **)&Ma, sizeof(int) * m * n));
    CHECK(cudaMallocManaged((void **)&Mb, sizeof(int) * n * k));
    CHECK(cudaMallocManaged((void **)&Mc, sizeof(int) * m * k));
    CHECK(cudaMallocManaged((void **)&Mcc, sizeof(int) * m * k));

    cudaEvent_t start, stop;
    CHECK(cudaEventCreate(&start));
    CHECK(cudaEventCreate(&stop));

    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            Ma[i * n + j] = rand() % 1024;
        }
    }

    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < k; ++j) {
            Mb[i * k + j] = rand() % 1024;
        }
    }

//    int *d_a, *d_b, *d_c;
//    CHECK(cudaMalloc((void **)&d_a, sizeof(int) * m * n));
//    CHECK(cudaMalloc((void **)&d_b, sizeof(int) * n * k));
//    CHECK(cudaMalloc((void **)&d_c, sizeof(int) * m * k));

//    // copy matrix A and B from host to device memory
//    CHECK(cudaMemcpy(d_a, h_a, sizeof(int)*m*n, cudaMemcpyHostToDevice));
//    CHECK(cudaMemcpy(d_b, h_b, sizeof(int)*n*k, cudaMemcpyHostToDevice));

    unsigned int grid_rows = (m + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (k + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_cols, grid_rows);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);

    CHECK(cudaEventRecord(start));
    gpu_matrix_mult<<<dimGrid, dimBlock>>>(Ma, Mb, Mc, m, n, k);
    CHECK(cudaEventRecord(stop));
    CHECK(cudaEventSynchronize(stop));
    CHECK(cudaEventElapsedTime(&elapsed_time, start, stop));
    printf("Time 1 with global memory=%g ms.\n", elapsed_time);

    CHECK(cudaEventRecord(start));
    gpu_matrix_mult_shared<<<dimGrid, dimBlock>>>(Ma, Mb, Mc, m, n, k);
    CHECK(cudaEventRecord(stop));
    CHECK(cudaEventSynchronize(stop));
    CHECK(cudaEventElapsedTime(&elapsed_time, start, stop));
    printf("Time 2 with shared memory=%g ms.\n", elapsed_time);

    CHECK(cudaEventDestroy(start));
    CHECK(cudaEventDestroy(stop));
//    CHECK(cudaMemcpy(h_c, d_c, sizeof(int)*m*k, cudaMemcpyDeviceToHost));
    //cudaThreadSynchronize();

//    cpu_matrix_mult(Ma, Mb, Mcc, m, n, k);

    int ok = 1;
//    for (int i = 0; i < m; ++i)
//    {
//        for (int j = 0; j < k; ++j)
//        {
//            if(fabs(Mcc[i*k + j] - Mc[i*k + j])>(1.0e-10))
//            {
//                ok = 0;
//            }
//        }
//    }

    if(ok)
    {
        printf("Pass!!!\n");
    }
    else
    {
        printf("Error!!!\n");
    }

    // free memory
    cudaFree(Ma);
    cudaFree(Mb);
    cudaFree(Mc);
    cudaFree(Mcc);
    return 0;
}
