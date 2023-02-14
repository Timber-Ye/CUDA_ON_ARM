//
// Created by hanchiao on 2/9/23.
//
#include <cstdio>
#include <cmath>
#include "../matmul/error.cuh"

#define N 5
#define BLOCK_SIZE 32

void print_matrix(int R, int C, int* A, const char* name)
{
    printf("%s = \n", name);
    for (int r = 0; r < R; ++r)
    {
        for (int c = 0; c < C; ++c)
        {
            printf("%5d", A[c * R + r]);
        }
        printf("\n");
    }
}

__global__ void transpose_global_memory(int * src, int *dst, int m, int n){
    int x = threadIdx.x + blockDim.x * blockIdx.x;
    int y = threadIdx.y + blockDim.y * blockIdx.y;

    if(x < n && y < m){
        dst[x + y*m] = src[y + x*n];
    }
}

__global__ void transpose_shared_memory(int * src, int m, int n){
    int x = threadIdx.x + blockDim.x * blockIdx.x;
    int y = threadIdx.y + blockDim.y * blockIdx.y;

    __shared__ int tile_s[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ int tile_d[BLOCK_SIZE][BLOCK_SIZE];

    if(blockIdx.x < blockIdx.y){
        int dx = threadIdx.x + blockDim.x * blockIdx.y;
        int dy = threadIdx.y + blockDim.y * blockIdx.x;
        if(x < m && y < n){
            tile_s[threadIdx.y][threadIdx.x] = src[x + y * n];
        }
        if(dx < m && dy <n){
            tile_d[threadIdx.y][threadIdx.x] = src[dx + dy * n];
        }
        __syncthreads();

        if(x < m && y < n){
            src[x + y * n] = tile_d[threadIdx.x][threadIdx.y];
        }
        if(dx < m && dy < n){
            src[dx + dy * n] = tile_s[threadIdx.x][threadIdx.y];
        }
        __syncthreads();

    }else if(blockIdx.x == blockIdx.y){
        if(x < m && y < n){
            tile_s[threadIdx.y][threadIdx.x] = src[x + y * n];
        }
        __syncthreads();

        if(x < m && y < n){
            src[x + y * n] = tile_s[threadIdx.x][threadIdx.y];
        }
        __syncthreads();
    }
}


int main(){
    int *h_a, *h_aT, *h_aTT, *d_a, *d_aT;
    CHECK(cudaMallocHost((void **)&h_a, sizeof(int) * N * N));
    CHECK(cudaMallocHost((void **)&h_aT, sizeof(int) * N * N));
    CHECK(cudaMallocHost((void **)&h_aTT, sizeof(int) * N * N));

    CHECK(cudaMalloc((void **)&d_a, sizeof(int) * N * N));
    CHECK(cudaMalloc((void **)&d_aT, sizeof(int) * N * N));

    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            h_a[i * N + j] = rand() % 32;
        }
    }
//    print_matrix(M, N, h_a, "A");

    CHECK(cudaMemcpy(d_a, h_a, sizeof(int) * N * N, cudaMemcpyHostToDevice));

    cudaEvent_t start, stop;
    float elapsed_time;
    CHECK(cudaEventCreate(&start));
    CHECK(cudaEventCreate(&stop));

    unsigned int grid_rows = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_cols, grid_rows);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);

    CHECK(cudaEventRecord(start));
    transpose_global_memory<<<dimGrid, dimBlock>>>(d_a, d_aT, N, N);
    CHECK(cudaEventRecord(stop));
    CHECK(cudaEventSynchronize(stop));
    CHECK(cudaEventElapsedTime(&elapsed_time, start, stop));

    CHECK(cudaEventDestroy(start));
    CHECK(cudaEventDestroy(stop));

    CHECK(cudaMemcpy(h_aT, d_aT, sizeof(int)*N*N, cudaMemcpyDeviceToHost));

//    print_matrix(M, N, h_aT, "A'");

    for(int i=0; i<N; i++){
        for(int j=0; j<N; j++){
            if(h_a[i+j*N]!=h_aT[j+i*N]){
                printf("Validation Failed!\n");
                return 0;
            }
        }
    }
    printf("Time 1 with global memory=%g ms.\n", elapsed_time);

    print_matrix(N, N, h_a, "A");
    CHECK(cudaMemcpy(d_a, h_a, sizeof(int) * N * N, cudaMemcpyHostToDevice));
    CHECK(cudaEventCreate(&start));
    CHECK(cudaEventCreate(&stop));

    CHECK(cudaEventRecord(start));
    transpose_shared_memory<<<dimGrid, dimBlock>>>(d_a, N, N);
    CHECK(cudaEventRecord(stop));
    CHECK(cudaEventSynchronize(stop));
    CHECK(cudaEventElapsedTime(&elapsed_time, start, stop));

    CHECK(cudaEventDestroy(start));
    CHECK(cudaEventDestroy(stop));

    CHECK(cudaMemcpy(h_aTT, d_a, sizeof(int)*N*N, cudaMemcpyDeviceToHost));

    print_matrix(N, N, h_aTT, "A'");
    for(int i=0; i<N; i++){
        for(int j=0; j<N; j++){
            if(h_a[i+j*N]!=h_aTT[j+i*N]){
                printf("Validation Failed!\n");
                return 0;
            }
        }
    }
    printf("Time 2 with shared memory=%g ms.\n", elapsed_time);


    CHECK(cudaFreeHost(h_a));
    CHECK(cudaFreeHost(h_aT));
    CHECK(cudaFreeHost(h_aTT));
    CHECK(cudaFree(d_a));
    CHECK(cudaFree(d_aT));
    return 0;
}