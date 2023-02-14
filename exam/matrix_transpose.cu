//
// Created by hanchiao on 2/11/23.
//
#include <stdio.h>
#include <stdlib.h>
#include "../matmul/error.cuh"

#define TILE_DIM 32   //Don't ask me why I don't set these two values to one
#define BLOCK_SIZE 32
#define N 3000 // for huanhuan, you know that!

__managed__ int input_M[N * N];      //input matrix & GPU result
int cpu_result[N * N];   //CPU result


//in-place matrix transpose
__global__ void ip_transpose(int* data)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;

    __shared__ int tile_I[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ int tile_II[BLOCK_SIZE][BLOCK_SIZE];

    if(blockIdx.x < blockIdx.y){
        int dx = threadIdx.x + blockIdx.y * blockDim.x;  // 对称块
        int dy = threadIdx.y + blockIdx.x * blockDim.y;

        if(x < N && y < N){
            tile_I[threadIdx.y][threadIdx.x] = data[y * N + x];
        }
        if(dx < N && dy < N){
            tile_II[threadIdx.y][threadIdx.x] = data[dy * N + dx];
        }
        __syncthreads();

        if(x < N && y < N){
            data[y * N + x] = tile_II[threadIdx.x][threadIdx.y];
        }
        if(dx < N && dy < N){
            data[dy * N +dx] = tile_I[threadIdx.x][threadIdx.y];
        }
        __syncthreads();
    }else if(blockIdx.x == blockIdx.y){
        if(x < N && y < N){
            tile_I[threadIdx.y][threadIdx.x] = data[y * N + x];
        }
        __syncthreads();

        if(x < N && y < N){
            data[y * N + x] = tile_I[threadIdx.x][threadIdx.y];
        }
        __syncthreads();
    }
//    int x = threadIdx.x + blockDim.x * blockIdx.x;
//    int y = threadIdx.y + blockDim.y * blockIdx.y;
//
//    __shared__ int tile_s[BLOCK_SIZE][BLOCK_SIZE];
//    __shared__ int tile_d[BLOCK_SIZE][BLOCK_SIZE];
//
//    if(blockIdx.x < blockIdx.y){
//        int dx = threadIdx.x + blockDim.x * blockIdx.y;
//        int dy = threadIdx.y + blockDim.y * blockIdx.x;
//        if(x < N && y < N){
//            tile_s[threadIdx.y][threadIdx.x] = data[x + y * N];
//        }
//        if(dx < N && dy <N){
//            tile_d[threadIdx.y][threadIdx.x] = data[dx + dy * N];
//        }
//        __syncthreads();
//
//        if(x < N && y < N){
//            data[x + y * N] = tile_d[threadIdx.x][threadIdx.y];
//        }
//        if(dx < N && dy < N){
//            data[dx + dy * N] = tile_s[threadIdx.x][threadIdx.y];
//        }
//        __syncthreads();
//
//    }else if(blockIdx.x == blockIdx.y){
//        if(x < N && y < N){
//            tile_s[threadIdx.y][threadIdx.x] = data[x + y * N];
//        }
//        __syncthreads();
//
//        if(x < N && y < N){
//            data[x + y * N] = tile_s[threadIdx.x][threadIdx.y];
//        }
//        __syncthreads();
//    }
}

void cpu_transpose(int* A, int* B)
{
    for (int j = 0; j < N; j++)
    {
        for (int i = 0; i < N; i++)
        {
            B[i * N + j] = A[j * N + i];
        }
    }
}

int main(int argc, char const* argv[])
{

    cudaEvent_t start, stop_gpu;
    CHECK(cudaEventCreate(&start));
    CHECK(cudaEventCreate(&stop_gpu));


    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            input_M[i * N + j] = rand() % 1000;
        }
    }
    cpu_transpose(input_M, cpu_result);

    CHECK(cudaEventRecord(start));
    unsigned int grid_rows = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_cols, grid_rows);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    ip_transpose <<<dimGrid, dimBlock >>> (input_M);
    CHECK(cudaDeviceSynchronize());
    CHECK(cudaEventRecord(stop_gpu));
    CHECK(cudaEventSynchronize(stop_gpu));

    float elapsed_time_gpu;
    CHECK(cudaEventElapsedTime(&elapsed_time_gpu, start, stop_gpu));
    printf("Time_GPU = %g ms.\n", elapsed_time_gpu);

    CHECK(cudaEventDestroy(start));
    CHECK(cudaEventDestroy(stop_gpu));

    int ok = 1;
    for (int i = 0; i < N; ++i)
    {
        for (int j = 0; j < N; ++j)
        {
            if (fabs(input_M[i * N + j] - cpu_result[i * N + j]) > (1.0e-10))
            {
                ok = 0;
            }
        }
    }


    if (ok)
    {
        printf("Pass!!!\n");
    }
    else
    {
        printf("Error!!!\n");
    }

    return 0;
}