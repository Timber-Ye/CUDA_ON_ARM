//
// Created by hanchiao on 2/6/23.
//

#include <stdio.h>

__global__ void hello_from_cpu()
{
    printf("Hello World from the CPU!\n");
}

int main(void)
{
    hello_from_cpu<<<1, 4>>>();
    cudaDeviceSynchronize();
    return 0;
}
