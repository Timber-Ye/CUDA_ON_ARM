//
// Created by hanchiao on 2/9/23.
//
#include <cstdio>
#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <sys/time.h>
#include <ctime>

#define N 5

int source[N];               //input data

void _init(int *ptr, int count)
{
    uint32_t seed = (uint32_t)time(NULL); //make huan happy
    srand(seed);  //reseeding the random generator

    //filling the buffer with random data
    for (int i = 0; i < count; i++) ptr[i] = rand();
}

int main(){
    _init(source, N);

    std::make_heap(source, source+3);

    for(int i=0; i<3; i++){
        printf("%d\n", source[i]);
    }
}

