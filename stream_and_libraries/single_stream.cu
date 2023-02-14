//
// Created by hanchiao on 2/9/23.
//
#include <cstdio>
#include "../matmul/error.cuh"

#define N (1024*1024)
#define FULL_DATA_SIZE   (N*20)

__global__ void kernel( int *a, int *b, int *c ) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < N) {
        int idx1 = (idx + 1) % 256;
        int idx2 = (idx + 2) % 256;
        float   as = (a[idx] + a[idx1] + a[idx2]) / 3.0f;
        float   bs = (b[idx] + b[idx1] + b[idx2]) / 3.0f;
        c[idx] = (as + bs) / 2;
    }
}

int main(void){
    cudaDeviceProp  prop;
    int whichDevice;
    CHECK( cudaGetDevice( &whichDevice ) );
    CHECK( cudaGetDeviceProperties( &prop, whichDevice ) );
    if (!prop.deviceOverlap) {
        printf( "Device will not handle overlaps, so no speed up from streams\n" );
        return 0;
    }

    int *h_a, *h_b, *h_c;
    int *d_a_0, *d_b_0, *d_c_0;

    cudaStream_t stream_0;
    CHECK(cudaStreamCreate(&stream_0));

    cudaEvent_t     start, stop;
    float           elapsedTime;
    CHECK(cudaEventCreate(&start));
    CHECK(cudaEventCreate(&stop));

    CHECK(cudaHostAlloc((void **)&h_a, sizeof(int) * FULL_DATA_SIZE, cudaHostAllocDefault));
    CHECK(cudaHostAlloc((void **)&h_b, sizeof(int) * FULL_DATA_SIZE, cudaHostAllocDefault));
    CHECK(cudaHostAlloc((void **)&h_c, sizeof(int) * FULL_DATA_SIZE, cudaHostAllocDefault));

    CHECK(cudaMalloc((void **)&d_a_0, sizeof(int) * N));
    CHECK(cudaMalloc((void **)&d_b_0, sizeof(int) * N));
    CHECK(cudaMalloc((void **)&d_c_0, sizeof(int) * N));

    for (int i=0; i<FULL_DATA_SIZE; i++) {
        h_a[i] = rand();
        h_b[i] = rand();
    }

    CHECK(cudaEventRecord(start, 0));
    for(int i=0; i<FULL_DATA_SIZE; i+= N){
        CHECK(cudaMemcpyAsync(d_a_0, h_a+i, sizeof(int) * N, cudaMemcpyHostToDevice, stream_0));

        CHECK(cudaMemcpyAsync(d_b_0, h_b+i, sizeof(int) * N, cudaMemcpyHostToDevice, stream_0));

        kernel<<<N/256,1024,0,stream_0>>>( d_a_0, d_b_0, d_c_0 );

        CHECK(cudaMemcpyAsync(h_c+i, d_c_0, N * sizeof(int), cudaMemcpyDeviceToHost, stream_0));
    }

    CHECK(cudaStreamSynchronize(stream_0));
    CHECK(cudaEventRecord(stop, 0));
    CHECK(cudaEventSynchronize(stop));
    CHECK(cudaEventElapsedTime(&elapsedTime, start, stop));

    printf("Time consuming: %3.1f ms\n", elapsedTime);

    // cleanup the streams and memory
    CHECK( cudaFreeHost( h_a ) );
    CHECK( cudaFreeHost( h_b ) );
    CHECK( cudaFreeHost( h_c ) );
    CHECK( cudaFree( d_a_0 ) );
    CHECK( cudaFree( d_b_0 ) );
    CHECK( cudaFree( d_c_0 ) );
    CHECK( cudaStreamDestroy( stream_0 ) );

    return 0;
}
