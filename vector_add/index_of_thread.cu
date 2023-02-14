//
// Created by hanchiao on 2/6/23.
//

#include <stdio.h>

__global__ void index_of_thread(){
    const int bid = blockIdx.x;
    const int tid = threadIdx.x;

    printf("Hello World from block %d and thread %d!\n", bid, tid);
}


int main(void)
{
    index_of_thread<<<5, 5>>>();
    cudaDeviceSynchronize();
    return 0;
}
