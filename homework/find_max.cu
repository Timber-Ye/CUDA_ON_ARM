//
// Created by hanchiao on 2/9/23.
//
#include <cstdint>
#include <cstdlib>
#include <sys/time.h>
#include <ctime>
#include "../matmul/error.cuh"

#define BLOCKSPERGRID 256
#define THREADSPERBLOCK 32
#define N 10000000

__managed__ int source[N];               //input data
__managed__ int final_result[1] = {INT_MIN};   //scalar output

int _find_max_cpu(int *ptr, int count)
{
    int max = INT_MIN;
    for (int i = 0; i < count; i++)
    {
        if(ptr[i] > max){
            max = ptr[i];
        }
    }
    return max;
}

__global__ void _find_max_gpu(int *ptr, int count, int *result){
    __shared__ int sum_per_block[THREADSPERBLOCK];

    int tmp = INT_MIN;
    for(int idx = threadIdx.x + blockIdx.x * blockDim.x; idx < count; idx += gridDim.x * blockDim.x){
        if(tmp < ptr[idx]) tmp = ptr[idx];
    }

    sum_per_block[threadIdx.x] = tmp;
    __syncthreads();

    for(int length = THREADSPERBLOCK / 2; length > 0; length /= 2){
        if(threadIdx.x < length){
            int max = sum_per_block[threadIdx.x] > sum_per_block[threadIdx.x + length] ? sum_per_block[threadIdx.x] : sum_per_block[threadIdx.x + length];
//            __syncthreads();
            sum_per_block[threadIdx.x] = max;
            __syncthreads();
        }
    }

    if(threadIdx.x == 0) atomicMax(result, sum_per_block[0]);
}

void _init(int *ptr, int count)
{
    uint32_t seed = (uint32_t)time(NULL); //make huan happy
    srand(seed);  //reseeding the random generator

    //filling the buffer with random data
    for (int i = 0; i < count; i++) ptr[i] = rand();
}

double get_time()
{
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return ((double)tv.tv_usec * 0.000001 + tv.tv_sec);
}

int main(){
    //**********************************
    fprintf(stderr, "filling the buffer with %d elements...\n", N);
    _init(source, N);

    //**********************************
    //Now we are going to kick off your kernel.
    cudaDeviceSynchronize(); //steady! ready! go!

    fprintf(stderr, "Running on GPU...\n");

    double t0 = get_time();
    _find_max_gpu<<<BLOCKSPERGRID, THREADSPERBLOCK>>>(source, N, final_result);
    CHECK(cudaGetLastError());  //checking for launch failures
    CHECK(cudaDeviceSynchronize()); //checking for run-time failures
    double t1 = get_time();

    int A = final_result[0];
    fprintf(stderr, "GPU max: %u\n", A);


    //**********************************
    //Now we are going to exercise your CPU...
    fprintf(stderr, "Running on CPU...\n");

    double t2 = get_time();
    int B = _find_max_cpu(source, N);
    double t3 = get_time();
    fprintf(stderr, "CPU max: %u\n", B);

    //******The last judgement**********
    if (A == B)
    {
        fprintf(stderr, "Test Passed!\n");
    }
    else
    {
        fprintf(stderr, "Test failed!\n");
        exit(-1);
    }

    //****and some timing details*******
    fprintf(stderr, "GPU time %.3f ms\n", (t1 - t0) * 1000.0);
    fprintf(stderr, "CPU time %.3f ms\n", (t3 - t2) * 1000.0);

    return 0;
}